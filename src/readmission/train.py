from __future__ import annotations
import json
import os
from pathlib import Path
from typing import List, Tuple, Optional

import joblib
import pandas as pd


def _detect_label_column(df: pd.DataFrame, explicit: Optional[str] = None) -> str:
    if explicit:
        if explicit in df.columns:
            return explicit
        for c in df.columns:
            if c.lower() == explicit.lower():
                return c
        raise ValueError(f"LABEL_COL='{explicit}' not found in columns: {list(df.columns)}")
    aliases = [
        'readmitted', 'readmission', 'readmission_30d', 'readmitted_30d',
        'is_readmitted', 'readmit', 'readmit_30d', 'readmit_flag', 'target',
    ]
    for a in aliases:
        if a in df.columns:
            return a
    lower_map = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a.lower() in lower_map:
            return lower_map[a.lower()]
    for c in df.columns:
        if 'readmit' in c.lower():
            return c
    raise ValueError("Could not find a label column. Set LABEL_COL env var or rename your target.")


def prepare_features(
    claims: pd.DataFrame,
    patients: pd.DataFrame,
    label_col: Optional[str] = None,
) -> Tuple[Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series], List[str]]:
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    label_col = label_col or os.getenv('LABEL_COL')
    label_col = _detect_label_column(claims, label_col)

    if not pd.api.types.is_numeric_dtype(claims[label_col]):
        m = {'yes':1,'y':1,'true':1,'t':1,'1':1,
             'no':0,'n':0,'false':0,'f':0,'0':0}
        claims[label_col] = (
            claims[label_col].astype(str).str.strip().str.lower().map(m).astype('float')
        )

    _date_candidates = [c for c in ['discharge_date','index_discharge_date','encounter_end','claim_end_date','service_to'] if c in claims.columns]
    if _date_candidates:
        _dcol = _date_candidates[0]
        claims[_dcol] = pd.to_datetime(claims[_dcol], errors='coerce')
        _max_date = claims[_dcol].max()
        _followup = pd.Timedelta(days=30)
        claims['label_eligible'] = claims[_dcol] <= (_max_date - _followup)
    else:
        claims['label_eligible'] = True

    claims['readmitted_missing'] = claims[label_col].isna()
    labeled = claims[claims['label_eligible'] & ~claims['readmitted_missing']].copy()

    patients_clean = patients.copy()
    id_cols = [c for c in ['patient_id','member_id','person_id'] if c in patients_clean.columns]
    col_nulls = patients_clean.isna().sum().rename('n_missing').to_frame()
    col_nulls['pct_missing'] = 100 * col_nulls['n_missing'] / max(len(patients_clean), 1)
    high_missing = [c for c, pct in col_nulls['pct_missing'].items() if pct > 60.0 and c not in id_cols]
    if high_missing:
        patients_clean.drop(columns=high_missing, inplace=True)
    cat_cols = [c for c in patients_clean.select_dtypes(include=['object','category']).columns if c not in id_cols]
    num_cols = [c for c in patients_clean.select_dtypes(include=['number']).columns if c not in id_cols]
    for c in cat_cols:
        patients_clean[f"{c}_was_missing"] = patients_clean[c].isna().astype('int8')
        patients_clean[c] = patients_clean[c].fillna('Unknown')
    for c in num_cols:
        patients_clean[f"{c}_was_missing"] = patients_clean[c].isna().astype('int8')
        med = patients_clean[c].median()
        patients_clean[c] = patients_clean[c].fillna(med)

    key_candidates = ['patient_id','member_id','person_id']
    keys = [k for k in key_candidates if (k in labeled.columns) and (k in patients_clean.columns)]
    if keys:
        key = keys[0]
        data = labeled.merge(patients_clean, on=key, how='left', suffixes=('', '_pt'))
        groups = data[key]
    else:
        data = labeled.copy()
        groups = None

    id_like = [c for c in data.columns if 'id' in c.lower()]
    date_like = [c for c in data.columns if 'date' in c.lower()]
    exclude = set([label_col] + id_like + date_like)
    X = data[[c for c in data.columns if c not in exclude]].copy()
    y = data[label_col].astype(int)

    if groups is not None and len(groups.unique()) > 1:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        train_idx, valid_idx = next(gss.split(X, y, groups=groups))
        X_tr, X_va = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[valid_idx]
    else:
        X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    return (X_tr, y_tr, X_va, y_va), X.columns.tolist()


def build_pipeline():
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.ensemble import HistGradientBoostingClassifier

    def make_preprocess(df: pd.DataFrame):
        num_cols = df.select_dtypes(include=['number', 'bool']).columns.tolist()
        cat_cols = df.select_dtypes(exclude=['number', 'bool']).columns.tolist()
        return ColumnTransformer([
            ('num', StandardScaler(with_mean=False), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=10), cat_cols),
        ], remainder='drop')

    model = HistGradientBoostingClassifier(
        learning_rate=0.1,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=0.0,
        max_depth=None,
        random_state=42,
    )
    return make_preprocess, model


def train_and_save(data_dir: str = 'data', artifacts_dir: str = 'artifacts'):
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import roc_auc_score, average_precision_score

    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
    claims = pd.read_csv(Path(data_dir) / 'claims.csv')
    patients = pd.read_csv(Path(data_dir) / 'patients.csv')
    (X_tr, y_tr, X_va, y_va), feat_names = prepare_features(claims, patients, label_col=os.getenv('LABEL_COL'))

    preprocess_factory, model = build_pipeline()
    preprocess = preprocess_factory(X_tr)
    pipe = Pipeline([
        ('prep', preprocess),
        ('model', model)
    ])
    pipe.fit(X_tr, y_tr)
    pr = average_precision_score(y_va, pipe.predict_proba(X_va)[:,1])
    auc = roc_auc_score(y_va, pipe.predict_proba(X_va)[:,1])
    print({'valid_pr_auc': round(pr,3), 'valid_roc_auc': round(auc,3)})

    bundle = {'pipeline': pipe, 'feature_names': list(X_tr.columns)}
    joblib.dump(bundle, Path(artifacts_dir) / 'model.joblib')
    model_card = {
        'metrics': {'valid_pr_auc': float(pr), 'valid_roc_auc': float(auc)},
        'prevalence_valid': float(y_va.mean()),
        'n_train': int(len(X_tr)),
        'n_valid': int(len(X_va)),
        'label_col': os.getenv('LABEL_COL') or 'auto-detected',
        'generated_at': __import__('datetime').datetime.utcnow().isoformat() + 'Z'
    }
    (Path(artifacts_dir) / 'model_card.json').write_text(json.dumps(model_card, indent=2))
    print(f"Saved model to {artifacts_dir}/model.joblib")


if __name__ == '__main__':
    train_and_save()
