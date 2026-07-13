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
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
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
    """Avalia UMA configuração por CV, sem jamais tocar o conjunto de teste.

    Usa `cross_validate` com múltiplas métricas numa única passada. Duas chamadas
    separadas a `cross_val_score` (uma para F1, outra para recall) treinariam os
    mesmos 5 folds duas vezes — o dobro do custo para o mesmo resultado.
    """
    cv = cv or make_cv()
    pipe = build_pipeline(estimator, escalonar=escalonar, winsorizar=winsorizar)

    scores = cross_validate(
        pipe, X_train, y_train, cv=cv,
        scoring={"f1": "f1", "recall": "recall"},
        n_jobs=-1,
    )
    f1_cv = scores["test_f1"].mean()
    recall_cv = scores["test_recall"].mean()

    pipe.fit(X_train, y_train)
    f1_train = f1_score(y_train, pipe.predict(X_train))  # treino REAL, não balanceado

    return ModelResult(modelo_nome, parametro, valor, f1_train, f1_cv, recall_cv)


def run_grid(
    X_train,
    y_train,
    ks: list[int] | None = None,
    depths: list | None = None,
    cv: StratifiedKFold | None = None,
) -> pd.DataFrame:
    """Roda o grid obrigatório: 4 valores de K e 4 de max_depth.

    `ks` e `depths` são `None` por padrão, e não listas literais: um argumento
    mutável como default é avaliado UMA vez, na definição da função, e passa a
    ser compartilhado entre todas as chamadas.
    """
    cv = cv or make_cv()
    ks = ks or [3, 5, 7, 9]
    depths = depths or [3, 5, 7, None]
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


def out_of_fold_proba(estimator, X_train, y_train, escalonar: bool,
                      winsorizar: bool, cv: StratifiedKFold | None = None) -> np.ndarray:
    """Probabilidades out-of-fold para o conjunto de TREINO.

    Cada observação recebe a probabilidade prevista por um modelo que NÃO a viu
    durante o treinamento — é o fold em que ela era validação. O resultado é uma
    estimativa honesta da probabilidade "fora da amostra", obtida sem gastar um
    único registro do conjunto de teste.

    Serve para calibrar decisões que dependem da probabilidade — notadamente a
    escolha do threshold (ver `evaluation.BusinessCost.threshold_from_oof`).
    Escolher o threshold contra `y_test` seria *selection leakage*: o mesmo erro
    que a seleção de hiperparâmetros pelo teste comete.
    """
    cv = cv or make_cv()
    pipe = build_pipeline(estimator, escalonar=escalonar, winsorizar=winsorizar)
    return cross_val_predict(
        pipe, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
