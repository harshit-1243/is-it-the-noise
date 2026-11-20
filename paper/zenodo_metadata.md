# Zenodo upload: metadata

**Resource type:** Publication / Preprint
**Title:** Is It the Noise? Decomposing DP-SGD's Membership Protection and Auditing Data-Level Defences on a Small Clinical Dataset
**Author:** Modi, Harshit. K. J. Somaiya School of Engineering, Somaiya Vidyavihar University (add your ORCID if you have one)
**Licence (suggested):** CC BY 4.0 for the paper. Give the code its own licence (e.g. MIT) in the code upload.
**Language:** English

## Description (150 words)

Small clinical prediction models can reveal who was in their training data. We audit diabetes classifiers trained on 192 records of the Pima Indians Diabetes dataset with a loss-threshold attack, a shadow-model attack and LiRA with 64 reference models. LiRA identifies 56.2% of a Random Forest's training patients at a 1% false-positive rate. Matched-recipe ablations split DP-SGD's protection into shorter training, per-record clipping and noise. Under the shadow attack the noise adds nothing measurable (p = 0.85), while under LiRA it accounts for 21% of the drop in attack AUC and lowers exposure at 1% FPR from 2.0% to 1.2%. Under the same adversary, Mondrian k-anonymity and compressive projections leave model leakage largely intact, and full-record local DP destroys utility. DP-SGD logistic regression at ε = 4 matches non-private accuracy. On 253,680 CDC BRFSS records the noise costs at most 0.003 ROC-AUC from 25,000 records. Code and logs reproduce every number.

## Keywords

membership inference; differential privacy; DP-SGD; LiRA; privacy auditing; k-anonymity; local differential privacy; compressive privacy; clinical machine learning; diabetes prediction

## Related identifiers

- "Is supplemented by": the Zenodo record or GitHub release of the code (src/, tests/, results/, run_all.sh)
