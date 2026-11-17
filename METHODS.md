# Method choices and justifications

Every method in this pipeline was selected against at least one concrete
alternative, on one of three grounds: **it was tested and measured better**,
**it is the established standard for this exact problem in the literature**, or
**the problem structure requires it**. This document records, stage by stage,
what was considered, what was chosen, and why — so that a reviewer can audit
each decision rather than take it on trust.

> **Provenance.** Stages 01, 03, 04 and 05 and the SoVI prioritisation are the
> defended Master's thesis (ENSA Khouribga, June 2026, graded 19/20). Stage 02
> (the ML prediction layer) was implemented after the defence, as the first
> step towards the proposed PhD work package WP1; in the thesis, demand was
> aggregated deterministically and road states were classified manually from a
> seismic study. Stage 02 replaces exactly those two manual steps — nothing
> else in the pipeline changed.

---

## Stage 01 — Clustering of beneficiary points: K-Means

**Role.** Group the 65 geocoded beneficiary points (schools of Agadir
Ida-Outanane) into zones that the location stage can serve.

**Alternatives tested.** DBSCAN (implemented in the same notebook), and
K-Means with K = 2…13.

**Why K-Means (K = 6).**
- *Tested:* the K sweep was scored by silhouette coefficient and inertia
  gain; K = 6 gave compact, balanced clusters (sizes 12/8/10/6/9/15) at the
  elbow of the curve.
- *Problem structure:* DBSCAN marks low-density points as noise and leaves
  them unassigned. In disaster response an unassigned community is an
  unserved community, so a partitioning method that assigns **every** point
  is required. K-Means also gives a controllable cluster count, which the
  downstream p-median stage needs.

## Stage 01b — Facility location: p-median

**Role.** Select 16 aid delivery points (ADPs) from the candidate points.

**Alternatives considered.** p-center (minimise the worst-case distance),
maximal covering location problem (maximise population within a radius).

**Why p-median.**
- *Literature standard:* the p-median is the reference model for locating
  relief facilities in humanitarian logistics (Balcik & Beamon, 2008), where
  the objective — minimise total demand-weighted travel distance — matches
  the operational goal of reaching the most people fastest on aggregate.
- *Problem structure:* p-center optimises for the single worst-served point
  and sacrifices the majority; covering models need an arbitrary service
  radius. With scarce vehicles, aggregate efficiency (p-median) plus an
  explicit fairness correction (SoVI, below) is the better decomposition.

## Stage 02a — Demand prediction: Random Forest + Gradient Boosting

**Role.** Predict each ADP's demand *before* operations, replacing the
thesis's after-the-fact aggregation of assigned beneficiary points.

**Alternatives tested.** The deterministic aggregation used in the thesis
(the "counting" baseline), a mean predictor, linear models.

**Why tree ensembles.**
- *Measured:* leave-one-out cross-validation on the 16 ADPs — Random Forest
  MAE 53.0, Gradient Boosting MAE 53.1, against 63.1 for the mean baseline
  (≈16% error reduction on a deliberately hard, tiny dataset).
- *Problem structure:* demand depends non-linearly on spatial and assignment
  features (distance to depot, cluster, assigned-point counts); tree
  ensembles capture this without distributional assumptions, are robust at
  very small n, and need no feature scaling. This is the same reason they are
  the default family in the ML literature on disaster-demand estimation.
- Both models are kept: their agreement is itself a stability signal, and
  the conformal layer (next) is wrapped around the Random Forest.

## Stage 02b — Uncertainty quantification: jackknife+ conformal prediction

**Role.** Turn the point demand forecast into an interval with a
finite-sample coverage guarantee — the uncertainty set the robust optimiser
of the proposed PhD (WP2) will consume.

**Alternatives considered.** Split (inductive) conformal prediction;
parametric prediction intervals from the forest's tree variance.

**Why jackknife+ (Barber, Candès, Ramdas & Tibshirani, 2021).**
- *Problem structure:* with n = 16, split conformal would sacrifice half
  the data to a calibration set. Jackknife+ is the conformal variant
  designed for very small samples — it uses every point for both fitting and
  calibration and still guarantees coverage.
- *Measured:* 90% intervals achieve 100% empirical coverage (conservative,
  as expected at this n) with mean width ≈ 197 demand units against a mean
  demand of 134 — an honest quantification of how uncertain demand really
  is, and precisely the quantity a robust plan must hedge against.
- *Lineage:* conformal uncertainty sets for robust optimisation follow
  Johnstone & Cox (2021); distribution-free uncertainty quantification
  follows Angelopoulos & Bates (2023).

## Stage 02c — Road-state classification: Random Forest + Gradient Boosting

**Role.** Predict each road segment's post-earthquake state (Open / Slow /
Blocked) from its OpenStreetMap attributes, replacing the thesis's manual
classification from a seismic study.

**Alternatives considered.** The manual rule-based classification used in
the thesis (reproducibility and speed limits); a single classifier.

**Why RF + GB on OSM attributes.**
- *Measured:* 5-fold cross-validation on 679 labelled segments from the
  thesis's T6 scenario — Gradient Boosting accuracy 0.90, Random Forest
  0.83 (macro F1 0.59 / 0.61); Random Forest recalls 86% of **blocked**
  segments, the safety-critical class.
- *Interpretability:* feature importances recover the physically sensible
  story — fragile rural classes (track, unclassified, unpaved surfaces)
  dominate the blocked set, while engineered classes (tertiary, secondary,
  asphalt) survive. The model agrees with domain intuition, which is what
  makes it auditable by a non-ML decision-maker.

## Stage 03 — Baseline routing: Google OR-Tools, Guided Local Search

**Role.** Solve the capacitated vehicle routing problem (CVRP) from the
depot (DEPOT_18) over the beneficiary points: 4 vehicles × capacity 20,
17 points, total load 65, total time 479 min.

**Alternatives considered.** Exact MIP solving (CPLEX-class); pure
metaheuristics (tabu search, genetic algorithms); the custom heuristic of
Stage 04.

**Why OR-Tools with PATH_CHEAPEST_ARC + Guided Local Search.**
- *Literature/practice standard:* OR-Tools is the reference open-source
  routing solver; Guided Local Search (Voudouris & Tsang, 1999) is its
  strongest general-purpose metaheuristic for CVRP, escaping local optima
  by penalising features of incumbent solutions.
- *Measured:* the script reproduces the thesis results **exactly**
  (re-run November 2026), which is the reproducibility standard the rest of
  the pipeline is held to.
- *Why not exact MIP:* CVRP is NP-hard; at operational scale the binding
  constraint is solution time during a crisis, not a certificate of
  optimality — a position standard in the disaster-operations literature
  (Altay & Green, 2006).

## Stage 04 — Heterogeneous-fleet routing: custom insertion + 2-opt heuristic

**Role.** Route the real fleet — 6 vehicles of three types with
post-earthquake capacity factors β = 0.7–0.8 — over the 16 ADPs
(total load 2,145; utilisation up to 99.8%).

**Alternatives considered.** Forcing this variant into OR-Tools with
workarounds; a general metaheuristic (ALNS) from scratch.

**Why a purpose-built heuristic.**
- *Problem structure:* the fleet model is non-standard (vehicle-dependent
  useful capacities degraded by the quake, farthest-seed initialisation to
  spread routes geographically). A transparent constructive heuristic —
  cheapest insertion with 2-opt local improvement — implements exactly
  these semantics with no solver workarounds, and every step is auditable,
  which matters when routes must be justified to field coordinators.
- *Measured:* reproduces the thesis output file exactly on re-run.
- The proposed PhD (WP2) revisits this stage with stability-penalised ALNS
  warm starts for re-planning — the heuristic is the baseline it must beat.

## Stage 05 — Disrupted-network re-planning (T6): OR-Tools, per-vehicle capacities

**Role.** Re-plan on the damaged network (≈65% of roads open, 25% slow,
10% blocked): 14 of 16 ADPs remain reachable; total time 556 min.

**Why this design.** The scenario structure (open/slow/blocked) is taken
from the seismic study of the thesis; the re-plan keeps the same solver
family as the baseline so that T0-vs-T6 differences reflect the *network*,
not the solver. Stage 02c exists precisely to make this stage's input
learned and reproducible instead of manual.

## Cross-cutting — Prioritisation under partial coverage: SoVI

**Role.** The Social Vulnerability Index is the pipeline's fairness layer.
Whenever the network cannot serve everyone — at T6, coverage falls to
92.2% of demand — the optimiser alone would choose *whom to abandon* by
cost. SoVI replaces that silent, cost-driven choice with an explicit,
published equity criterion: serve in order of social vulnerability.

**Construction.** Following Cutter, Boruff & Shirley (2003), socio-economic
indicators of each community (demographic structure, housing and living
conditions, access to services — the indicators available for the study
communes) are standardised and combined into a single vulnerability score
per ADP. Higher score = a population less able to anticipate, cope with and
recover from the disaster on its own.

**Alternatives considered.**
- *Cost/distance only (the CVRP default):* efficient in aggregate, but it
  systematically drops the farthest points first — in practice the poorest,
  most rural communities, i.e. precisely the populations a relief operation
  exists to reach. Rejected on both ethical and operational grounds.
- *Population size only:* serves the largest points first, but size is not
  need — a large, well-connected urban point would outrank a small, highly
  vulnerable rural one. Rejected.
- *Manual or expert ranking:* defensible case-by-case but not reproducible
  and not auditable; two coordinators would rank differently. Rejected in
  favour of a standardised, published methodology.
- *Other composite indices* (hazard-specific exposure indices, ad-hoc local
  scores): SoVI is the reference index for social vulnerability to
  environmental hazards, replicated across many countries and hazards,
  which makes the scores comparable with the literature rather than a
  one-off local construction.

**Why SoVI, in this pipeline.**
- *It changes real decisions:* at T6, with ADP_7 and ADP_12 unreachable,
  the reinforcement vehicle (V6) is activated for **ADP_13** — not the
  cheapest unserved point, but the most vulnerable reachable one — and the
  remaining coverage is allocated by vulnerability rank.
- *It makes the equity criterion contestable:* a coordinator, a reviewer or
  an affected community can see *why* a point was prioritised, because the
  score and its inputs are explicit. An artefact of distance minimisation
  could not be defended in the same way.
- *It is the PhD's fairness anchor:* the proposed WP2 (robust re-planning)
  needs a criterion for what must be protected when plans degrade, and WP3
  (explanations) needs human-legible reasons — "ADP_13 was prioritised
  because its vulnerability score is the highest of the reachable points"
  is exactly the kind of reason SoVI supplies. Equity in humanitarian
  logistics is a first-class objective, not an afterthought (cf. the
  multi-objective humanitarian-response literature in *Transportation
  Research Part E*).

---

## References

- Altay, N. & Green, W.G. (2006). OR/MS research in disaster operations
  management. *European Journal of Operational Research*, 175(1).
- Angelopoulos, A.N. & Bates, S. (2023). Conformal prediction: a gentle
  introduction. *Foundations and Trends in Machine Learning*, 16(4).
- Balcik, B. & Beamon, B.M. (2008). Facility location in humanitarian
  relief. *International Journal of Logistics: Research and Applications*,
  11(2).
- Barber, R.F., Candès, E.J., Ramdas, A. & Tibshirani, R.J. (2021).
  Predictive inference with the jackknife+. *Annals of Statistics*, 49(1).
- Barber, R.F., Candès, E.J., Ramdas, A. & Tibshirani, R.J. (2023).
  Conformal prediction beyond exchangeability. *Annals of Statistics*,
  51(2).
- Bertsimas, D. & Sim, M. (2004). The price of robustness. *Operations
  Research*, 52(1).
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1).
- Cutter, S.L., Boruff, B.J. & Shirley, W.L. (2003). Social vulnerability
  to environmental hazards. *Social Science Quarterly*, 84(2).
- Elmachtoub, A.N. & Grigas, P. (2022). Smart "Predict, then Optimize".
  *Management Science*, 68(1).
- Ester, M., Kriegel, H.-P., Sander, J. & Xu, X. (1996). A density-based
  algorithm for discovering clusters (DBSCAN). *Proc. KDD-96*.
- Friedman, J.H. (2001). Greedy function approximation: a gradient
  boosting machine. *Annals of Statistics*, 29(5).
- Hakimi, S.L. (1964). Optimum locations of switching centers and the
  absolute centers and medians of a graph. *Operations Research*, 12(3).
- Johnstone, C. & Cox, B. (2021). Conformal uncertainty sets for robust
  optimization. *Proc. COPA / PMLR*, 152.
- Lykouris, T. & Vassilvitskii, S. (2021). Competitive caching with
  machine learned advice. *Journal of the ACM*, 68(4).
- Voudouris, C. & Tsang, E. (1999). Guided local search and its
  application to the traveling salesman problem. *European Journal of
  Operational Research*, 113(2).
