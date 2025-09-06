from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

_DOWNLOADED_PATH: Path | None = None
_bundle = None
_expected_features: Optional[List[str]] = None

app = FastAPI(title='Readmission Model API', version='0.1.0')
def _default_maps():
    categorical_defaults = {
        'gender': 'Unknown',
        'ethnicity': 'Unknown',
        'insurance_type': 'Unknown',
        'smoker': 'Unknown',
        'icd_code': 'Unknown',
    }
    numeric_defaults = {
        'age': 0,
        'bmi': 0.0,
        'cost': 0.0,
        'label_eligible': 1,
        'readmitted_missing': 0,
    }
    return categorical_defaults, numeric_defaults


MODEL_PATH_ENV = os.getenv('MODEL_PATH', 'artifacts/model.joblib')


def _resolve_model_path() -> Path:
    """Resolve MODEL_PATH; if it starts with gs://, download once to /tmp."""
    global _DOWNLOADED_PATH
    if MODEL_PATH_ENV.startswith('gs://'):
        if _DOWNLOADED_PATH is None:
            # Lazy import to avoid requiring GCS when not used
            from google.cloud import storage  # type: ignore
            import tempfile
            # Parse gs://bucket/blob
            _, path = MODEL_PATH_ENV.split('gs://', 1)
            bucket_name, *blob_parts = path.split('/')
            blob_name = '/'.join(blob_parts)
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            tmp_dir = Path(tempfile.gettempdir())
            local = tmp_dir / 'model.joblib'
            blob.download_to_filename(local)
            _DOWNLOADED_PATH = local
        return _DOWNLOADED_PATH
    else:
        return Path(MODEL_PATH_ENV)


def _get_pipeline():
    global _bundle, _expected_features
    if _bundle is None:
        resolved = _resolve_model_path()
        _bundle = joblib.load(resolved)
        # Expect training saved {'pipeline': ..., 'feature_names': [...]}
        ef = _bundle.get('feature_names')
        if ef is not None:
            if isinstance(ef, (pd.Index, pd.Series, tuple)):
                _expected_features = list(ef)
            elif isinstance(ef, str):
                _expected_features = [ef]
            else:
                _expected_features = list(ef)
    return _bundle['pipeline']


def _align_input(df: pd.DataFrame) -> pd.DataFrame:
    """Align incoming rows to the training feature schema by adding missing
    columns with safe defaults and ordering columns to expected order.
    """
    _ = _get_pipeline()  # ensures bundle and expected features are loaded
    global _expected_features
    if not _expected_features:
        # Fall back to pass-through
        return df

    exp = list(_expected_features)
    X = df.copy()

    # Defaults for known categoricals and internal flags
    categorical_defaults, numeric_defaults = _default_maps()

    for col in exp:
        if col in X.columns:
            continue
        if col.endswith('_was_missing'):
            X[col] = 0
        elif col in categorical_defaults:
            X[col] = categorical_defaults[col]
        elif col in numeric_defaults:
            X[col] = numeric_defaults[col]
        else:
            # Heuristic: treat as numeric by default
            X[col] = 0

    # Keep only expected columns and in expected order (ensure DataFrame result)
    X = X.loc[:, exp]
    if isinstance(X, pd.Series):
        X = X.to_frame().T
    return pd.DataFrame(X)


class ScoreRequest(BaseModel):
    rows: list[dict]


@app.get('/healthz')
async def healthz():
    return {'status': 'ok'}


@app.post('/score')
async def score(req: ScoreRequest) -> Dict[str, Any]:
    pipe = _get_pipeline()
    X_raw = pd.DataFrame(req.rows)
    X = _align_input(X_raw)
    probs = pipe.predict_proba(X)[:, 1]
    return {'probabilities': probs.tolist(), 'n': len(probs)}


@app.get('/metadata')
async def metadata() -> Dict[str, Any]:
    _ = _get_pipeline()
    cat_def, num_def = _default_maps()
    return {
        'expected_features': _expected_features,
        'defaults': {
            'categorical': cat_def,
            'numeric': num_def,
            'missing_flag_default': 0,
        },
        'model_path': MODEL_PATH_ENV,
        'version': '0.1.0',
    }
