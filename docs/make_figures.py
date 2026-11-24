"""Regenerate the measured figures in docs/figures/.

    uv run --locked --group notebook python docs/make_figures.py

Trains the selected pipeline with the package (same split and seed as
``customer_churn_ml.training``) into a temporary directory, then plots
holdout results. Also copies two figures that the analysis notebook already
rendered. ``architecture.svg`` is a hand-drawn illustrative diagram and is
not produced here.

Computation functions (``compute_*``) return data only; plotting functions
(``plot_*``) only render pre-computed data.
"""

from __future__ import annotations

import base64
import json
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from customer_churn_ml.evaluation import positive_class_proba  # noqa: E402
from customer_churn_ml.training import (  # noqa: E402
    DEFAULT_DATA_PATH,
    load_raw,
    run_training,
    split_features_target,
)

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"
NOTEBOOK = ROOT / "notebooks" / "churn_analysis.ipynb"

# Reference palette (validated): series slot 1 + neutral, text and surface tokens.
MODEL_COLOR = "#2a78d6"
BASELINE_COLOR = "#b5b4ae"
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e0"

# Notebook figures worth reusing: (cell id, image index within the cell, output name).
NOTEBOOK_FIGURES = [
    ("b459e8ba", 0, "hyperparameter_effects_ap.png"),
]


# ---------------------------------------------------------------- computation


def compute_holdout_scores(data_path: Path, model_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Train with the package and return holdout labels, scores and its metrics."""
    result = run_training(data_path, model_dir / "model.joblib", include_oof=False)
    X, y = split_features_target(load_raw(data_path))
    y_test = y.loc[result.test_index].to_numpy()
    scores = positive_class_proba(result.pipeline, X.loc[result.test_index])
    return y_test, scores, result.holdout_metrics


def compute_pr_curve(y_true: np.ndarray, scores: np.ndarray, k: int) -> dict:
    """Precision-recall curve plus the operating point of the top-K list."""
    precision, recall, _ = precision_recall_curve(y_true, scores)
    order = np.argsort(-scores, kind="stable")
    hits = int(y_true[order[:k]].sum())
    return {
        "precision": precision,
        "recall": recall,
        "prevalence": float(y_true.mean()),
        "k": k,
        "k_precision": hits / k,
        "k_recall": hits / int(y_true.sum()),
    }


def compute_topk_comparison(metrics: dict) -> dict:
    """Churners expected per K contacts: random selection vs. model ranking."""
    k = int(metrics["k"])
    return {
        "k": k,
        "random": metrics["prevalence"] * k,
        "model": metrics["precision_at_k"] * k,
        "lift": metrics["lift_at_k"],
    }


def extract_notebook_png(notebook: Path, cell_id: str, index: int) -> bytes:
    """PNG bytes of the index-th image output of a notebook cell."""
    nb = json.loads(notebook.read_text(encoding="utf-8"))
    cell = next(c for c in nb["cells"] if c.get("id") == cell_id)
    images = [o["data"]["image/png"] for o in cell.get("outputs", []) if "image/png" in o.get("data", {})]
    return base64.b64decode(images[index])


# ---------------------------------------------------------------- visualization


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_pr_curve(pr: dict, ap: float, ax) -> None:
    ax.plot(pr["recall"], pr["precision"], color=MODEL_COLOR, linewidth=2, drawstyle="steps-post")
    ax.axhline(pr["prevalence"], color=BASELINE_COLOR, linestyle="--", linewidth=1.5)
    ax.text(0.99, pr["prevalence"] + 0.02, f"Random ranking (churn rate {pr['prevalence']:.1%})",
            ha="right", va="bottom", fontsize=9, color=TEXT_SECONDARY)
    ax.plot(pr["k_recall"], pr["k_precision"], "o", markersize=8, color=MODEL_COLOR,
            markeredgecolor=SURFACE, markeredgewidth=2)
    ax.annotate(
        f"Top {pr['k']} customers\nprecision {pr['k_precision']:.2f}, recall {pr['k_recall']:.2f}",
        (pr["k_recall"], pr["k_precision"]), xytext=(18, 8), textcoords="offset points",
        fontsize=9, color=TEXT_PRIMARY,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall (share of all churners found)", color=TEXT_SECONDARY)
    ax.set_ylabel("Precision", color=TEXT_SECONDARY)
    ax.set_title(f"Holdout precision-recall curve · RandomForest · AP = {ap:.4f}",
                 loc="left", fontsize=11, color=TEXT_PRIMARY)
    _style(ax)


def plot_topk_comparison(cmp: dict, ax) -> None:
    labels = ["Random selection", "Model top-ranked"]
    values = [cmp["random"], cmp["model"]]
    bars = ax.barh(labels, values, color=[BASELINE_COLOR, MODEL_COLOR], height=0.5)
    for bar, value in zip(bars, values):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}" if value % 1 else f"{value:.0f}",
                va="center", fontsize=11, color=TEXT_PRIMARY)
    ax.set_xlim(0, cmp["k"])
    ax.invert_yaxis()
    ax.set_xlabel(f"Actual churners among {cmp['k']} contacted holdout customers", color=TEXT_SECONDARY)
    ax.set_title(f"Precision@{cmp['k']}: {cmp['lift']:.2f}× more churners reached than random",
                 loc="left", fontsize=11, color=TEXT_PRIMARY)
    _style(ax)
    ax.grid(axis="y", visible=False)


# ---------------------------------------------------------------- main


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        y_test, scores, metrics = compute_holdout_scores(DEFAULT_DATA_PATH, Path(tmp))

    pr = compute_pr_curve(y_test, scores, int(metrics["k"]))
    fig, ax = plt.subplots(figsize=(7, 4.6), facecolor=SURFACE)
    plot_pr_curve(pr, metrics["average_precision"], ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pr_curve_holdout.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 2.6), facecolor=SURFACE)
    plot_topk_comparison(compute_topk_comparison(metrics), ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "precision_at_100_vs_random.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    for cell_id, index, name in NOTEBOOK_FIGURES:
        (FIG_DIR / name).write_bytes(extract_notebook_png(NOTEBOOK, cell_id, index))

    print(json.dumps({k: metrics[k] for k in ("average_precision", "precision_at_k", "lift_at_k", "k")}))


if __name__ == "__main__":
    main()
