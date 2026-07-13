"""Constantes centrais do projeto.

Manter os nomes de colunas e as premissas de negócio em um único lugar evita
divergência silenciosa entre o notebook, os módulos e a documentação.
"""

from __future__ import annotations

RANDOM_STATE = 42
TARGET = "loan_status"

# ---------------------------------------------------------------- features
NUMERICAS = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "cb_person_cred_hist_length",
    "comprometimento_renda",
]

# Genuinamente nominais: não existe "aluguel < hipoteca".
NOMINAIS = ["person_home_ownership", "loan_intent"]

# Ordinal: A < B < ... < G. One-hot destruiria a ordem que carrega o sinal.
ORDINAL = ["loan_grade"]
GRADES = ["A", "B", "C", "D", "E", "F", "G"]

# Binária Y/N -> 0/1 (uma coluna, não duas).
BINARIA = ["cb_person_default_on_file"]
BINARIA_CATEGORIAS = ["N", "Y"]

# Colunas sujeitas a clipping — APENAS no ramo do KNN (ver decisoes-tecnicas.md §5).
WINSORIZAR = [
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "comprometimento_renda",
]

# Redundante com comprometimento_renda (r = 0.9989): sai da matriz de modelagem.
FEATURE_REDUNDANTE = "loan_percent_income"

# ---------------------------------------------------------- regras de negócio
IDADE_MINIMA = 18
IDADE_MAXIMA = 100
IDADE_MINIMA_TRABALHO = 14  # emp_length não pode exceder (idade - 14)

# ------------------------------------------------- premissas do custo (hipóteses)
LGD = 0.65  # Loss Given Default: fração do principal perdida no calote
DURACAO_MEDIA_ANOS = 2.0  # horizonte usado para a margem de juros perdida no FP
