# Stage 06 — SoVI-proxy prioritisation

The pipeline's fairness layer: a transparent, reproducible
vulnerability-prioritisation score for the 16 aid delivery points (ADPs),
in the spirit of the Social Vulnerability Index (SoVI; Cutter, Boruff &
Shirley, 2003).

## Why a proxy

The full SoVI is built from census socio-economic variables that are not
available at ADP level for the study communes. This stage therefore
computes a **SoVI proxy** from the variables the case study actually
measured — the name used in the thesis working files
(`ADP_SoVI_T6_proxy`). In the defended thesis, prioritisation under
partial coverage was argued qualitatively in the T6 analysis; this script
is its reproducible formalisation, built after the defence.

## Method

Four components, each z-standardised (higher = more vulnerable):

| Component | Variable | Rationale |
|---|---|---|
| Exposure | ADP demand | people depending on the point |
| Remoteness | road distance from DEPOT_18 | response delay |
| Isolation | mean distance to the beneficiary points it serves | last-mile difficulty |
| Disruption exposure | 1 if unreachable in the T6 network | network-failure exposure |

Composite = **unweighted mean** of the z-scores. Equal weights are
deliberate: with proxy variables and n = 16 there is no defensible basis
for differential weights, and equal weighting is auditable. Two
sensitivity variants are computed with the score (no-disruption variant;
PCA first component, Cutter et al.'s original construction) and their
rank agreement is reported.

## Results (verified run)

- Overall ranking is headed by **BP_12** (score 1.87) — high demand, most
  remote (67.9 km), and unreachable at T6.
- **T6 service priority among the 14 reachable ADPs: BP_13 first**
  (score 0.89; highest demand of all ADPs, 297; 53.3 km from the depot).
  This independently reproduces the defended thesis's T6 decision, where
  the reinforcement vehicle V6 was activated for ADP_13.
- Sensitivity: Spearman rank correlation of the main score with the
  no-disruption variant = **0.956**, with the PCA variant = **0.988** —
  the priority order is robust to the construction choice.

## Files

- `sovi_prioritisation.py` — the computation (run from this folder).
- `sovi_proxy_scores.csv` — components, composite score, overall rank,
  T6 reachability and T6 priority for all 16 ADPs.
