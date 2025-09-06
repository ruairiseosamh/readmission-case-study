from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import pandas as pd


def load_model(artifacts_dir: str = 'artifacts'):
    bundle = joblib.load(Path(artifacts_dir) / 'model.joblib')
    return bundle['pipeline']


def score_csv(input_csv: str, output_csv: str, artifacts_dir: str = 'artifacts'):
    pipe = load_model(artifacts_dir)
    df = pd.read_csv(input_csv)
    probs = pipe.predict_proba(df)[:, 1]
    out = df.copy()
    out['readmitted_proba'] = probs
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"Wrote scores to {output_csv}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--artifacts', default='artifacts')
    args = ap.parse_args()
    score_csv(args.input, args.output, artifacts_dir=args.artifacts)
