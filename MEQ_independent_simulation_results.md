# Independent simulation results for the MEQ follow-up

This is an exploratory independent reimplementation from the manuscript’s prose, not a reproduction of the unavailable `meq_pilot.py`. The simulation was not preregistered and its outputs must not be merged with the original pilot results. This run checks a matched-control design and identifies what a confirmatory run must fix.

## Run specification

- 10 paired seeds; 300 training episodes per factorial cell; 40 greedy evaluation episodes per seed and condition.
- Two safe start locations: (0,0) and (0,6).
- 2 × 2 factorial: interoceptive sensor vs fixed null readings crossed with drive-reduction vs task reward.
- Outcomes: thresholded viability fraction (steps inside the viability set / 200) and survival to step 200; nominal and 1.6× metabolism.
- Interventions on the trained sensor-plus-drive-reward policy: fixed-bin clamp and empirical joint sensor donor shuffle, with no retraining.
- All arms use a 3,136-state, 5-action Q table with the same learning rate, discount, and exploration schedule. Epsilon was rescaled from 1.0 to 0.05 over this short 300-episode debug run, unlike the manuscript’s 1,200-episode decay.
- Transition/event-order assumptions and unspecified implementation details are listed in [the follow-up protocol](MEQ_follow_up_protocol.md).

## Results

Table entries are seed means ± 95% t half-intervals (10 seeds; t(9) critical value 2.262).

| Start | Sensing / reward | Nominal viability | Shift viability | Nominal survival | Shift survival |
|---|---|---:|---:|---:|---:|
| 0,0 | Sensor + drive reward | 0.252 ± 0.014 | 0.159 ± 0.008 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| 0,0 | Null + drive reward | 0.245 ± 0.000 | 0.155 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| 0,0 | Sensor + task reward | 0.245 ± 0.000 | 0.155 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| 0,0 | Null + task reward | 0.245 ± 0.000 | 0.155 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| 0,6 | Sensor + drive reward | 0.281 ± 0.024 | 0.175 ± 0.014 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| 0,6 | Null + drive reward | 0.245 ± 0.000 | 0.155 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| 0,6 | Sensor + task reward | 0.265 ± 0.022 | 0.167 ± 0.014 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| 0,6 | Null + task reward | 0.258 ± 0.020 | 0.163 ± 0.012 | 0.000 ± 0.000 | 0.000 ± 0.000 |

No evaluation episode survived to step 200 across the 3,200 factorial evaluations. Policies therefore did not learn a viable behavior within this short budget. Any small viability contrasts are not evidence of MEQ efficacy or validation.

For the sensing effect under drive-reduction reward, paired sensing-minus-null viability was 0.007 (95% CI −0.007 to 0.021) at (0,0) and 0.036 (0.013 to 0.060) at (0,6). Under task reward, the corresponding differences were 0.000 at (0,0) and 0.006 (−0.020 to 0.033) at (0,6). The start-dependent pattern and zero survival preclude a pooled efficacy claim.

### Causal interventions

Intervention contrasts are paired intact-minus-intervention differences. Positive values favor intact sensing.

| Start | Contrast | Nominal viability difference | Shift viability difference | Survival difference |
|---|---|---:|---:|---:|
| 0,0 | Intact − Fixed clamp | 0.008 ± 0.014 | 0.004 ± 0.007 | 0.000 ± 0.000 |
| 0,0 | Intact − Donor shuffle | -0.017 ± 0.016 | -0.003 ± 0.008 | 0.000 ± 0.000 |
| 0,6 | Intact − Fixed clamp | 0.037 ± 0.024 | 0.019 ± 0.014 | 0.000 ± 0.000 |
| 0,6 | Intact − Donor shuffle | 0.001 ± 0.023 | 0.006 ± 0.014 | 0.000 ± 0.000 |

The fixed clamp reduced nominal viability by 0.008 (95% CI −0.005 to 0.022) at (0,0) and by 0.037 (0.013 to 0.060) at (0,6). Donor shuffling did not reduce viability: intact-minus-shuffle was −0.017 (−0.034 to −0.001) at (0,0) and 0.001 (−0.021 to 0.024) at (0,6). All survival contrasts were zero. The clamp result depends on start location and may include an unfamiliar-input effect. Because the underlying policies did not survive any evaluation episode, these interventions do **not** demonstrate meaningful channel dependence in a competent policy.

The per-seed normalized gate C = (P_intact − P_intervention) / P_intact was 0.029 ± 0.045 for the fixed clamp and −0.073 ± 0.064 for donor shuffle at (0,0); at (0,6), it was 0.120 ± 0.072 and −0.006 ± 0.078, respectively (mean ± 95% t half-interval). Estimates are start-sensitive and span zero. Since intact viability is low, C is only a descriptive diagnostic here.

## Interpretation and next step

The exploratory run does not validate the MEQ gate. The chosen short budget and assumed transition semantics did not produce competent policies. Before testing necessity, establish convergence with frozen mechanics and matched factorial controls. The highest-value next simulation step is the 2,000-episode, 30-seed prospective design in the protocol, using the manuscript’s 1,200-episode exploration decay and prespecified held-out starts and layouts. If policies remain nonviable, revise the simulator or training design before interpreting interventions.

## Reproducibility

- Source: [meq_independent_reimplementation.py](meq_independent_reimplementation.py); run `python outputs\meq_independent_reimplementation.py` from the project root (NumPy required).
- Protocol: [MEQ_follow_up_protocol.md](MEQ_follow_up_protocol.md).
- Per-seed factorial outcomes: [meq_independent_factorial_seed_results.csv](meq_independent_factorial_seed_results.csv).
- Per-seed intervention outcomes: [meq_independent_intervention_seed_results.csv](meq_independent_intervention_seed_results.csv).
- Machine-readable summaries: [meq_independent_summary.json](meq_independent_summary.json).
- Source SHA-256: `cc109e8aa640404251b8116708390bdaac8a70a7dc4b9c40109d925ae4789f98`.
