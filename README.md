# Is It the Noise? Decomposing DP-SGD's Membership Protection and Auditing Data-Level Defences on a Small Clinical Dataset

Code, results and paper source for the preprint by **Harshit Modi** (K. J. Somaiya School of Engineering, Somaiya Vidyavihar University).
Preprint PDF: [`paper/is_it_the_noise_preprint.pdf`](paper/is_it_the_noise_preprint.pdf)

**What it does.** Diabetes classifiers trained on 192 records of the Pima Indians Diabetes dataset are audited with three black-box membership inference attacks (loss threshold, 8-shadow-model attack, online LiRA with 64 models). Matched-recipe baselines split DP-SGD's protection into training length, per-record clipping and noise. The same adversary audits local DP, Mondrian k-anonymity and compressive projections (PCA, DCA, privacy-aware DCA). A scaling study on 253,680 CDC BRFSS 2015 records measures how the cost of DP noise changes with dataset size.

**Main findings.** LiRA identifies 56.2% of a Random Forest's training patients at 1% FPR. Under a shadow-model attack the noise step of DP-SGD is not significant (p = 0.85); under LiRA it accounts for 21% of the drop in attack AUC and lowers exposure at 1% FPR from 2.0% to 1.2%. DP-SGD logistic regression at ε = 4 matches non-private accuracy (0.756 vs 0.764, p = 0.41).

## Layout

| Path | Contents |
|---|---|
| `src/` | DP-SGD from scratch with Rényi-DP accounting, attacks (loss, shadow, LiRA), local DP, Mondrian k-anonymity, PCA/DCA/PDCA, seven experiment scripts |
| `tests/` | 13 correctness tests |
| `results/` | Logs, per-split runs and summaries (mean, 95% CI) for every experiment |
| `figures/` | Figures from the original experiment report |
| `paper/` | LaTeX source (IEEE), generated tables and figures, scripts for the LiRA bootstrap, figures, tables and claim checks |
| `data/` | Pima Indians Diabetes CSV; instructions for BRFSS 2015 |

## Reproduce

```bash
pip install -r requirements.txt
bash run_all.sh          # 13 tests + 7 experiments + figures; about 20-30 min on a 2-core CPU, no GPU
```

A full rerun in a fresh environment reproduces every file in `results/` exactly. The BRFSS scaling study needs the BRFSS file first (see `data/README.md`).

Paper numbers (from the repository root):

```bash
python -I paper/scripts/lira_bootstrap.py . paper/analysis                    # LiRA rerun + record-level bootstrap
python -I paper/scripts/extra_checks.py . paper/analysis/extra_checks.json
python -I paper/scripts/make_paper_figures.py . paper/analysis/lira_bootstrap.json paper/figures
python -I paper/scripts/make_paper_tables.py . paper/analysis/lira_bootstrap.json paper/analysis/extra_checks.json paper/tables
python -I paper/scripts/check_paper_claims.py . paper/analysis/lira_bootstrap.json paper/tables/derived.json paper/main.tex
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Data

- **Pima Indians Diabetes** (768 × 9), `data/pima_diabetes.csv`, SHA-256 `c980e5e01dc1c554a358102b773f34b29599d0051ec5cf5b56ad1c6fcb3a23b0`. NIDDK via the UCI repository; listed on Kaggle as CC0. The records come from a long-running study in the Gila River Indian Community; see the paper's ethics statement.
- **CDC Diabetes Health Indicators (BRFSS 2015)**, not included; see `data/README.md`.

No re-identification or linkage is attempted; attacks target only models trained here.

## Citation

```bibtex
@misc{modi2026noise,
  author = {Harshit Modi},
  title  = {Is It the Noise? Decomposing {DP-SGD}'s Membership Protection and Auditing Data-Level Defences on a Small Clinical Dataset},
  year   = {2026},
  note   = {Preprint}
}
```
