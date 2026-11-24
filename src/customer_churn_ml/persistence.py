"""Save and load the fitted churn Pipeline (cleaning + preprocessing + model).

Artifacts
---------
``<name>.joblib``  the complete fitted ``sklearn.pipeline.Pipeline``
``<name>.json``    sidecar: ``{"metadata": <caller metadata>, "environment": {...}}``

Security
--------
joblib files are pickles: loading one can execute arbitrary code. Only load
model files that you created yourself or that come from a trusted source.
Never call ``load_model`` on a file received from an untrusted user.
"""

from __future__ import annotations

import json
import os
import platform
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

MODEL_SUFFIX = ".joblib"


def metadata_path_for(model_path: str | Path) -> Path:
    """Path of the JSON sidecar that belongs to ``model_path``."""
    return Path(model_path).with_suffix(".json")


def _environment() -> dict[str, str]:
    """Library versions needed to load the pickle safely and reproduce results."""
    return {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "joblib": joblib.__version__,
    }


def _atomic_write(path: Path, write) -> None:
    """Write via a temp file in the same directory, then rename into place.

    Atomic for this single file only; it gives no guarantee across files.
    """
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        write(tmp)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def save_model(
    pipeline: Pipeline,
    path: str | Path,
    metadata: dict[str, Any] | None = None,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Save a fitted Pipeline with joblib plus a JSON metadata sidecar.

    Parameters
    ----------
    pipeline : fitted ``Pipeline`` (feature cleaning, preprocessing and model).
    path : destination ending in ``.joblib``; parent directories are created.
    metadata : JSON-serializable dict (e.g. hyperparameters, metrics, seed).
    overwrite : if False (default), raise ``FileExistsError`` when the model
        or its sidecar already exists.

    Returns
    -------
    (model_path, metadata_path)

    Notes
    -----
    Each file is written to a temp file and renamed into place, so neither
    file is ever left half-written. The two-file save as a whole is **not**
    atomic: the model is renamed first, then the sidecar. A crash between
    the two renames leaves a model without its sidecar (or, with
    ``overwrite=True``, a new model next to the previous sidecar). A
    retry with ``overwrite=False`` then raises ``FileExistsError``;
    inspect or delete the leftover file, or rerun with ``overwrite=True``.
    """
    model_path = Path(path)
    if model_path.suffix != MODEL_SUFFIX:
        raise ValueError(f"Model path must end with {MODEL_SUFFIX!r}: {model_path}")
    if not isinstance(pipeline, Pipeline):
        raise TypeError(f"Expected a sklearn Pipeline, got {type(pipeline).__name__}")
    check_is_fitted(pipeline)

    meta_path = metadata_path_for(model_path)
    if not overwrite:
        existing = [str(p) for p in (model_path, meta_path) if p.exists()]
        if existing:
            raise FileExistsError(
                f"Refusing to overwrite existing artifact(s): {existing}. "
                "Pass overwrite=True to replace them."
            )

    # Serialize metadata first so invalid metadata fails before anything is written.
    sidecar = json.dumps(
        {"metadata": metadata or {}, "environment": _environment()}, indent=2, sort_keys=True
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    # Per-file atomic only; see Notes above for the gap between these two renames.
    _atomic_write(model_path, lambda tmp: joblib.dump(pipeline, tmp))
    _atomic_write(meta_path, lambda tmp: Path(tmp).write_text(sidecar + "\n", encoding="utf-8"))
    return model_path, meta_path


def load_model(path: str | Path) -> Pipeline:
    """Load a Pipeline saved by ``save_model``.

    Only load files from trusted sources: joblib/pickle can execute arbitrary
    code during loading.
    """
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    pipeline = joblib.load(model_path)
    if not isinstance(pipeline, Pipeline):
        raise TypeError(
            f"{model_path} does not contain a sklearn Pipeline (got {type(pipeline).__name__})"
        )
    return pipeline


def load_metadata(path: str | Path) -> dict[str, Any]:
    """Read the JSON sidecar for a model path (``.joblib``) or sidecar path (``.json``)."""
    meta_path = metadata_path_for(path)
    if not meta_path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))
