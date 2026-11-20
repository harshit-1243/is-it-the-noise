# Data

`pima_diabetes.csv` is included (768 records, 8 features + `Outcome`).

The BRFSS 2015 file is not redistributed here. To run `src/exp_scaling.py`:

1. Download `diabetes_binary_health_indicators_BRFSS2015.csv` (CDC Diabetes Health Indicators, UCI Machine Learning Repository, DOI 10.24432/C53919; also on Kaggle as `alexteboul/diabetes-health-indicators-dataset`).
2. Compress it: `gzip -c diabetes_binary_health_indicators_BRFSS2015.csv > data/brfss2015_diabetes_binary.csv.gz`
3. The script checks the SHA-256 of the uncompressed CSV against `19f367e3e3350768f0c144c5d73ee5b355f67a57eaaa86ca7bd8aec594d8b1d0` and records the result in `results/exp_scaling/provenance_check.json`.
