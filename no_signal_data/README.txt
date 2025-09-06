Healthcare Big Case Study (Synthetic)
------------------------------------
Files:
 - patients.csv (≈15k rows + ~0.5% duplicates)
 - claims.csv   (≈250k rows + ~1% duplicates)

Purpose:
 - Practice end-to-end healthcare data science: cleaning, joining, feature engineering,
   modeling (e.g., 30-day readmission), evaluation, and insights.

Messiness injected intentionally:
 - Missing values (age, bmi, zip, icd_code, readmitted_30d)
 - Inconsistent encodings (gender variants, 'NULL' text, icd_code lowercase/trailing spaces)
 - Outliers (very high costs), anomalies (zero/negative cost)
 - Temporal issues (discharge before admit on some rows)
 - Duplicate rows

Suggested next steps:
 1) Clean & standardize fields; de-duplicate.
 2) Validate dates; compute length_of_stay; clip/winsorize costs.
 3) Map ICD codes to comorbidity categories; aggregate patient history.
 4) Train/evaluate readmission models (logistic/XGBoost) with AUROC/AUPRC.