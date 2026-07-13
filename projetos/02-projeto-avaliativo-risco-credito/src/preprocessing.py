"""Feature engineering, transformadores customizados e montagem dos pipelines.

REGRA DE OURO DESTE MÓDULO: nada aqui aprende parâmetro fora de um `fit`.
Medianas, percentis de corte, categorias e escalas são todos estimados no
`fit` — que o `Pipeline` do imbalanced-learn só executa sobre o fold de
treino. Isso torna o vazamento de dados *estruturalmente impossível*, em vez
de depender da disciplina de quem escreve o notebook.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .config import (
    BINARIA,
    BINARIA_CATEGORIAS,
    FEATURE_REDUNDANTE,
    GRADES,
    NOMINAIS,
    NUMERICAS,
    ORDINAL,
    RANDOM_STATE,
    WINSORIZAR,
)


# ------------------------------------------------------------------ features
def add_income_commitment(df: pd.DataFrame) -> pd.DataFrame:
    """Cria a feature obrigatória: % da renda anual comprometida com o empréstimo.

    A divisão por zero está estruturalmente impedida — `clean_basic_data` já
    garantiu `person_income > 0`. O `np.where` permanece como cinto de segurança.
    """
    out = df.copy()
    out["comprometimento_renda"] = np.where(
        out["person_income"].gt(0),
        (out["loan_amnt"] / out["person_income"]) * 100,
        np.nan,
    )
    return out


def drop_redundant_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Remove `loan_percent_income` da matriz de modelagem.

    Correlação de 0.9989 com `comprometimento_renda`: é a mesma informação em
    outra escala. Manter as duas daria peso DOBRADO a essa dimensão no cálculo
    de distância do KNN — um viés silencioso. A coluna permanece disponível na
    EDA; sai apenas da modelagem.
    """
    return df.drop(columns=[FEATURE_REDUNDANTE], errors="ignore")


# ---------------------------------------------------- transformadores customizados
class RateByGradeImputer(BaseEstimator, TransformerMixin):
    """Imputa `loan_int_rate` pela mediana do respectivo `loan_grade`.

    A taxa de juros é função quase determinística do rating do contrato. Uma
    mediana global achataria essa estrutura; a mediana por grade a preserva.
    O fallback para a mediana global cobre grades ausentes no treino.
    """

    def fit(self, X: pd.DataFrame, y=None):
        self.medianas_ = X.groupby("loan_grade")["loan_int_rate"].median()
        self.mediana_global_ = X["loan_int_rate"].median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        preenchimento = X["loan_grade"].map(self.medianas_)
        X["loan_int_rate"] = (
            X["loan_int_rate"].fillna(preenchimento).fillna(self.mediana_global_)
        )
        return X


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clipping no percentil `q`. Usado APENAS no ramo do KNN.

    Preserva a LINHA e achata apenas o VALOR extremo — ao contrário da remoção
    por IQR, que descartaria milhares de clientes de alta renda perfeitamente
    legítimos (justamente os bons pagadores que o banco não quer perder).

    A árvore não recebe este passo: seus cortes são ordinais e indiferentes a
    valores extremos. O KNN mede distância euclidiana e é dominado por eles.
    """

    def __init__(self, cols: list[str] | None = None, q: float = 0.99):
        self.cols = cols if cols is not None else WINSORIZAR
        self.q = q

    def fit(self, X: pd.DataFrame, y=None):
        self.limites_ = {c: X[c].quantile(self.q) for c in self.cols}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for coluna, limite in self.limites_.items():
            X[coluna] = X[coluna].clip(upper=limite)
        return X


# ------------------------------------------------------------------ pipelines
def build_preprocessor(escalonar: bool) -> ColumnTransformer:
    """Monta o pré-processamento.

    `escalonar=True`  -> ramo KNN (distância euclidiana exige escala comum).
    `escalonar=False` -> ramo Árvore (cortes ordinais dispensam escala; ver a
                         prova de invariância em `modeling.prove_scale_invariance`).
    """
    passos_num = [("imputer", SimpleImputer(strategy="median"))]
    passos_ord = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OrdinalEncoder(
                categories=[GRADES],
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
        ),
    ]
    if escalonar:
        passos_num.append(("scaler", StandardScaler()))
        passos_ord.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        [
            ("num", Pipeline(passos_num), NUMERICAS),
            (
                "nom",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                NOMINAIS,
            ),
            ("ord", Pipeline(passos_ord), ORDINAL),
            (
                "bin",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                categories=[BINARIA_CATEGORIAS],
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                BINARIA,
            ),
        ]
    )


def build_pipeline(
    modelo,
    escalonar: bool,
    winsorizar: bool,
    balancear: bool = True,
    balanceador=None,
) -> ImbPipeline:
    """Pipeline completo: imputação -> (clipping) -> encoding -> (balanceamento) -> modelo.

    O balanceador padrão é o `RandomUnderSampler`, e não o SMOTE, por dois motivos
    verificados empiricamente (ver decisoes-tecnicas.md §6):

    1. O SMOTE interpola linearmente e QUEBRA O ONE-HOT, gerando clientes
       sintéticos "71% proprietários e 29% inquilinos" — cortes que o modelo
       aprende e que nunca se repetem no teste.
    2. O SMOTE busca vizinhos por distância euclidiana, logo é SENSÍVEL À ESCALA.
       Isso destruiria a prova de invariância de escala da árvore: escalonar
       mudaria os dados sintéticos e, portanto, a árvore.

    O RandomUnderSampler não sintetiza nada, não calcula distância — e entregou
    o melhor recall da classe inadimplente na validação cruzada.
    """
    passos: list[tuple] = [("taxa_por_grade", RateByGradeImputer())]

    if winsorizar:
        passos.append(("winsorizer", Winsorizer()))

    passos.append(("preprocessador", build_preprocessor(escalonar)))

    if balancear:
        if balanceador is None:
            balanceador = RandomUnderSampler(random_state=RANDOM_STATE)
        passos.append(("balanceamento", balanceador))

    passos.append(("modelo", modelo))
    return ImbPipeline(passos)
