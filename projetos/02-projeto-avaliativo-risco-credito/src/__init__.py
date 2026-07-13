"""Pipeline preditivo de risco de crédito — SCTEC/SENAI.

API pública dos módulos:

    config        constantes, listas de features e premissas de custo
    data_quality  carga, auditoria de regras de negócio e limpeza
    preprocessing feature engineering, transformadores e montagem dos pipelines
    modeling      grid por validação cruzada e diagnóstico de overfitting
    evaluation    custo de negócio, threshold ótimo e análise de sensibilidade
"""

from . import config, data_quality, evaluation, modeling, preprocessing
from .config import RANDOM_STATE, TARGET
from .data_quality import (
    build_quality_report,
    clean_basic_data,
    load_data,
    validate_business_rules,
)
from .evaluation import BusinessCost, confusion_counts, plot_confusion_matrices
from .modeling import (
    make_cv,
    out_of_fold_proba,
    prove_scale_invariance,
    run_grid,
    select_best,
)
from .preprocessing import (
    RateByGradeImputer,
    Winsorizer,
    add_income_commitment,
    build_pipeline,
    build_preprocessor,
    drop_redundant_feature,
)

__all__ = [
    "config", "data_quality", "preprocessing", "modeling", "evaluation",
    "RANDOM_STATE", "TARGET",
    "load_data", "build_quality_report", "validate_business_rules", "clean_basic_data",
    "add_income_commitment", "drop_redundant_feature",
    "RateByGradeImputer", "Winsorizer", "build_preprocessor", "build_pipeline",
    "run_grid", "select_best", "prove_scale_invariance", "make_cv", "out_of_fold_proba",
    "BusinessCost", "plot_confusion_matrices", "confusion_counts",
]

__version__ = "1.0.0"
