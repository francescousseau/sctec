"""Otimização de hiperparâmetros e diagnóstico de overfitting.

Duas correções metodológicas estão codificadas aqui — e são o coração do
projeto:

(a) SELEÇÃO POR VALIDAÇÃO CRUZADA NO TREINO. Escolher `k` ou `max_depth`
    olhando a performance no teste é *selection leakage*: o holdout deixa de
    ser honesto e vira um conjunto de validação disfarçado. O teste só é
    aberto na fase final, uma única vez.

(b) O GAP É MEDIDO NO TREINO REAL, NÃO NO BALANCEADO. O F1 depende da
    prevalência da classe positiva. Comparar o F1 de um treino balanceado
    (50% de positivos) com o F1 do teste (22%) produz um "gap" que é, em boa
    parte, artefato de proporção — e não overfitting. Aqui, `f1_train` é
    calculado sobre a distribuição original.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from .config import RANDOM_STATE
from .preprocessing import build_pipeline


def make_cv(n_splits: int = 5) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


@dataclass
class ModelResult:
    modelo: str
    parametro: str
    valor: object
    f1_train: float
    f1_cv: float
    recall_cv: float

    @property
    def gap(self) -> float:
        """Overfitting real: F1 no treino (distribuição original) - F1 na validação."""
        return self.f1_train - self.f1_cv

    def to_dict(self) -> dict:
        return {
            "modelo": self.modelo,
            "parametro": self.parametro,
            "valor": self.valor,
            "f1_train": self.f1_train,
            "f1_cv": self.f1_cv,
            "recall_cv": self.recall_cv,
            "gap": self.gap,
        }


def evaluate_config(
    modelo_nome: str,
    parametro: str,
    valor,
    estimator,
    X_train,
    y_train,
    escalonar: bool,
    winsorizar: bool,
    cv: StratifiedKFold | None = None,
) -> ModelResult:
    """Avalia UMA configuração por CV, sem jamais tocar o conjunto de teste."""
    cv = cv or make_cv()
    pipe = build_pipeline(estimator, escalonar=escalonar, winsorizar=winsorizar)

    f1_cv = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1").mean()
    recall_cv = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="recall").mean()

    pipe.fit(X_train, y_train)
    f1_train = f1_score(y_train, pipe.predict(X_train))  # treino REAL, não balanceado

    return ModelResult(modelo_nome, parametro, valor, f1_train, f1_cv, recall_cv)


def run_grid(
    X_train,
    y_train,
    ks: list[int] = [3, 5, 7, 9],
    depths: list = [3, 5, 7, None],
    cv: StratifiedKFold | None = None,
) -> pd.DataFrame:
    """Roda o grid obrigatório: 4 valores de K e 4 de max_depth."""
    cv = cv or make_cv()
    resultados: list[ModelResult] = []

    for k in ks:
        resultados.append(
            evaluate_config(
                "KNN", f"k={k}", k,
                KNeighborsClassifier(n_neighbors=k),
                X_train, y_train,
                escalonar=True, winsorizar=True,  # KNN: escala + clipping
                cv=cv,
            )
        )

    for depth in depths:
        resultados.append(
            evaluate_config(
                "Árvore", f"max_depth={depth}", depth,
                DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE),
                X_train, y_train,
                escalonar=False, winsorizar=False,  # Árvore: nem escala, nem clipping
                cv=cv,
            )
        )

    return pd.DataFrame([r.to_dict() for r in resultados])


def select_best(tabela: pd.DataFrame, modelo: str, criterio: str = "f1_cv"):
    """Seleciona a melhor configuração de um modelo — pela CV, nunca pelo teste."""
    linha = tabela[tabela.modelo == modelo].sort_values(criterio, ascending=False).iloc[0]
    valor = linha["valor"]
    return None if pd.isna(valor) else valor, linha


def prove_scale_invariance(X_train, y_train, X_test, max_depth: int = 5) -> float:
    """Prova empírica de que a Árvore dispensa escalonamento.

    Treina a MESMA árvore com e sem StandardScaler e compara as predições.
    Retorna a concordância — que deve ser exatamente 1.0.

    Por quê: a árvore só pergunta "renda > X?". O StandardScaler é uma
    transformação monotônica — não reordena valores. Logo, os mesmos cortes
    são encontrados. Já o KNN mede distância euclidiana: sem escala,
    `person_income` (~10^4) esmagaria `person_age` (~10^1) e a vizinhança
    seria decidida praticamente só pela renda.

    Nota: o balanceador PRECISA ser livre de distância para que esta prova
    funcione. Com SMOTE, escalonar mudaria os dados sintéticos e a concordância
    cairia para ~92% — um dos motivos da escolha do RandomUnderSampler.
    """
    sem = build_pipeline(
        DecisionTreeClassifier(max_depth=max_depth, random_state=RANDOM_STATE),
        escalonar=False, winsorizar=False,
    ).fit(X_train, y_train).predict(X_test)

    com = build_pipeline(
        DecisionTreeClassifier(max_depth=max_depth, random_state=RANDOM_STATE),
        escalonar=True, winsorizar=False,
    ).fit(X_train, y_train).predict(X_test)

    return float(np.mean(sem == com))
