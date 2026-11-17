# Stage 02 — ML prediction layer

This stage is the pipeline's prediction layer. It was implemented after the
defence of the Master's thesis (June 2026), as a first step towards the proposed
PhD project *"Uncertainty-Aware Learning-Augmented Optimisation for
Decision-Making in Crisis Logistics"* (LJMU VC PhD Scholarship application,
2027 entry).

## Why this stage exists

In the thesis, two components were computed deterministically:

- **ADP demand** was counted (aggregation of assigned beneficiary points), not predicted.
- **Road states** in the disrupted (T6) scenario were classified manually from a seismic study.

The PhD replaces both with a *prediction layer* whose uncertainty is quantified
and passed to the optimiser (WP1: conformal calibration; WP2: robust
location-routing). The scripts here build that layer's first prototype on the
thesis's own case-study data.

## Contents

| Script | What it does | Headline result |
|---|---|---|
| `demand_prediction.py` | Random Forest + Gradient Boosting regression of ADP demand from spatial/assignment features | LOOCV MAE ≈ 53 (vs 63 for the mean baseline), n = 16 ADPs |
| `conformal_prediction.py` | Jackknife+ conformal intervals (Barber et al., 2021) around the RF demand model | 90% intervals, empirical coverage 100% (conservative at n = 16), mean width ≈ 197 |
| `road_state_classification.py` | Random Forest + Gradient Boosting classification of road segments as Open / Slow / Blocked from OpenStreetMap attributes, trained on the thesis's T6 labels | 5-fold CV accuracy 0.83 (RF) / 0.90 (GB), macro F1 0.61 / 0.59 |

All results are deliberately reported as proofs of concept: the case study is
small (16 ADPs; 679 labelled road segments), and the PhD will scale the
approach to synthetic and semi-synthetic scenario ensembles, exactly as set
out in work package WP1 of the PhD proposal.

## Run

```bash
pip install -r ../requirements.txt
python demand_prediction.py
python conformal_prediction.py
python road_state_classification.py
```
