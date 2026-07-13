"""Carga, auditoria e limpeza da base.

Princípio: separar *outlier estatístico* de *erro de cadastro*. Um cliente que
ganha muito é um outlier legítimo e deve ser preservado; um cliente de 144 anos
é dado sujo e deve ser corrigido. A confusão entre os dois leva a jogar fora
justamente os bons pagadores de alta renda.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    IDADE_MAXIMA,
    IDADE_MINIMA,
    IDADE_MINIMA_TRABALHO,
)


def load_data(path: str | Path) -> pd.DataFrame:
    """Carrega o CSV validando a existência do arquivo."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Base não encontrada em: {path.resolve()}")
    return pd.read_csv(path)


def build_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Relatório de tipos, ausências e cardinalidade."""
    return pd.DataFrame(
        {
            "tipo": df.dtypes.astype(str),
            "nulos": df.isna().sum(),
            "%_nulos": (df.isna().mean() * 100).round(2),
            "valores_unicos": df.nunique(dropna=False),
        }
    ).sort_values("%_nulos", ascending=False)


def validate_business_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Conta violações de regras lógicas SEM alterar a base.

    Roda antes da limpeza, para que a decisão seja tomada com o número na mesa.
    """
    violacoes = {
        f"idade fora de {IDADE_MINIMA}-{IDADE_MAXIMA} anos": (
            ~df["person_age"].between(IDADE_MINIMA, IDADE_MAXIMA)
        ).sum(),
        f"tempo de emprego > (idade - {IDADE_MINIMA_TRABALHO})": (
            df["person_emp_length"].notna()
            & (df["person_emp_length"] > df["person_age"] - IDADE_MINIMA_TRABALHO)
        ).sum(),
        "renda <= 0": df["person_income"].le(0).sum(),
        "empréstimo <= 0": df["loan_amnt"].le(0).sum(),
        "duplicatas completas": df.duplicated().sum(),
    }
    return pd.DataFrame.from_dict(violacoes, orient="index", columns=["registros"])


def clean_basic_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Limpeza determinística, com rastreabilidade de cada decisão.

    Ordem importa:

    1. Duplicatas saem ANTES do split. Se uma linha e sua cópia caíssem em lados
       opostos da separação treino/teste, o modelo teria visto o gabarito da
       prova durante o estudo — vazamento clássico.
    2. Idade impossível -> remove a LINHA (o cadastro inteiro é suspeito).
    3. Tempo de emprego impossível -> anula só o CAMPO (a linha é aproveitável;
       o valor será imputado depois, dentro do pipeline, com estatística de treino).
    4. Renda/empréstimo não positivos -> remove (inviabilizam a feature de
       comprometimento de renda).
    """
    log: list[str] = []
    n_inicial = len(df)

    limpo = df.drop_duplicates().copy()
    log.append(f"Duplicatas removidas: {n_inicial - len(limpo)}")

    antes = len(limpo)
    limpo = limpo[limpo["person_age"].between(IDADE_MINIMA, IDADE_MAXIMA)].copy()
    log.append(f"Idades impossíveis removidas: {antes - len(limpo)}")

    emp_invalido = limpo["person_emp_length"].notna() & (
        limpo["person_emp_length"] > limpo["person_age"] - IDADE_MINIMA_TRABALHO
    )
    log.append(f"Tempos de emprego convertidos em NaN: {int(emp_invalido.sum())}")
    limpo.loc[emp_invalido, "person_emp_length"] = np.nan

    antes = len(limpo)
    limpo = limpo[limpo["person_income"].gt(0) & limpo["loan_amnt"].gt(0)].copy()
    log.append(f"Renda/empréstimo não positivos removidos: {antes - len(limpo)}")

    log.append(f"{n_inicial:,} -> {len(limpo):,} linhas (retenção: {len(limpo)/n_inicial:.2%})")

    if verbose:
        print("\n".join(log))

    return limpo
