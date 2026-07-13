"""Avaliação sob a ótica do negócio: reais, não percentuais.

A decisão central deste módulo é NÃO usar custos fixos e arbitrários
(FP = R$ 500, FN = R$ 5.000). Errar num empréstimo de R$ 35 mil não custa o
mesmo que errar num de R$ 1 mil. O custo é modelado sobre o valor real de
cada contrato do conjunto avaliado.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from .config import DURACAO_MEDIA_ANOS, LGD


class BusinessCost:
    """Modelo de custo dos erros, calibrado pelo valor de cada empréstimo.

    Falso Negativo (emprestou a quem não pagou):
        perda = loan_amnt * LGD
        O banco perde o PRINCIPAL. LGD (Loss Given Default) = 65% por padrão:
        parte é recuperada via cobrança.

    Falso Positivo (negou crédito a quem pagaria):
        perda = loan_amnt * taxa_juros * duracao
        O banco perde a MARGEM, não o capital.

    Ambos são HIPÓTESES DECLARADAS — por isso `sensitivity_analysis` existe.
    """

    def __init__(
        self,
        loan_amnt: pd.Series | np.ndarray,
        loan_int_rate: pd.Series | np.ndarray,
        lgd: float = LGD,
        duracao_anos: float = DURACAO_MEDIA_ANOS,
    ):
        self.valores = np.asarray(loan_amnt, dtype=float)
        taxas = pd.Series(loan_int_rate, dtype=float)
        self.taxas = (taxas.fillna(taxas.median()).to_numpy()) / 100
        self.lgd = lgd
        self.duracao = duracao_anos

    def compute(self, y_true, y_pred, lgd: float | None = None) -> dict:
        lgd = self.lgd if lgd is None else lgd
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        mask_fp = (y_true == 0) & (y_pred == 1)
        mask_fn = (y_true == 1) & (y_pred == 0)

        custo_fp = float((self.valores[mask_fp] * self.taxas[mask_fp] * self.duracao).sum())
        custo_fn = float((self.valores[mask_fn] * lgd).sum())

        return {
            "FP": int(mask_fp.sum()),
            "FN": int(mask_fn.sum()),
            "custo_FP (margem perdida)": custo_fp,
            "custo_FN (calote)": custo_fn,
            "CUSTO TOTAL": custo_fp + custo_fn,
        }

    def baseline_aprovar_todos(self, y_true) -> dict:
        """Cenário sem modelo: o banco aprova todo mundo."""
        return self.compute(y_true, np.zeros(len(y_true), dtype=int))

    # ------------------------------------------------------------- threshold
    def optimal_threshold(
        self, y_true, y_proba, grid: np.ndarray | None = None
    ) -> tuple[float, list[float], np.ndarray]:
        """Varre o threshold e devolve o que MINIMIZA o custo total.

        O corte de 0.5 não é sagrado: ele é o ótimo apenas quando FP e FN custam
        a mesma coisa — o que quase nunca é verdade em crédito.
        """
        grid = np.arange(0.05, 0.96, 0.01) if grid is None else grid
        custos = [self.compute(y_true, (y_proba >= t).astype(int))["CUSTO TOTAL"] for t in grid]
        return float(grid[int(np.argmin(custos))]), custos, grid

    def plot_threshold_curve(self, y_true, probas: dict, output_path=None) -> None:
        plt.figure(figsize=(9, 5))
        for nome, proba in probas.items():
            t_otimo, custos, grid = self.optimal_threshold(y_true, proba)
            plt.plot(grid, custos, lw=2, label=nome)
            plt.scatter([t_otimo], [min(custos)], s=90, zorder=5)
            plt.annotate(
                f"t* = {t_otimo:.2f}\nR$ {min(custos):,.0f}",
                (t_otimo, min(custos)),
                textcoords="offset points", xytext=(10, 12), fontsize=9,
            )
        plt.axvline(0.5, ls="--", c="gray", label="threshold padrão (0.5)")
        plt.title("Custo total do erro × threshold de decisão")
        plt.xlabel("Threshold")
        plt.ylabel("Custo total (R$)")
        plt.legend()
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.show()

    # --------------------------------------------------------- sensibilidade
    def sensitivity_analysis(
        self, y_true, preds: dict, lgds: list[float] | None = None
    ) -> pd.DataFrame:
        """O veredito sobrevive a premissas diferentes?

        Se o modelo vencedor muda quando o LGD muda, a recomendação é frágil e
        depende de ter acertado um chute. Se não muda, é robusta.
        """
        lgds = lgds or [0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
        linhas = []
        for lgd in lgds:
            custos = {
                nome: self.compute(y_true, pred, lgd=lgd)["CUSTO TOTAL"]
                for nome, pred in preds.items()
            }
            vencedor = min(custos, key=custos.get)
            linhas.append({"LGD": lgd, **custos, "vencedor": vencedor})
        return pd.DataFrame(linhas)


# ---------------------------------------------------------------- matrizes
def plot_confusion_matrices(y_true, preds: dict, output_path=None) -> None:
    """Matrizes lado a lado. O canto inferior esquerdo é o FALSO NEGATIVO."""
    fig, axes = plt.subplots(1, len(preds), figsize=(6 * len(preds), 5))
    axes = np.atleast_1d(axes)

    for ax, (titulo, pred) in zip(axes, preds.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_true, pred,
            display_labels=["Bom pagador", "Inadimplente"],
            values_format="d", cmap="Blues", ax=ax, colorbar=False,
        )
        ax.set_title(titulo)

    plt.suptitle(
        "Matrizes de confusão — canto inferior esquerdo = FALSO NEGATIVO (o erro caro)",
        y=1.02,
    )
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def confusion_counts(y_true, y_pred) -> dict[str, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
