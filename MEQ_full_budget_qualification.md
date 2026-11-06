# Full budget simulator qualification

This independent reimplementation was trained for 2,000 episodes using the manuscript’s epsilon schedule (1.0 to 0.05 over the first 1,200 episodes). Five training seeds per start were evaluated at checkpoints using 20 greedy episodes per seed and metabolism condition. The mechanics are those explicitly recorded in the independent implementation; this is not a reproduction of the unavailable original source.

## Checkpoint results

Cells are training-seed means ± 95% t half-intervals (n = 5; t(4) critical value 2.776). The small seed count makes these screening estimates.

| Start | Metabolism | Episode | Viability fraction | Survival rate |
|---|---:|---:|---:|---:|
| 0,0 | 1.0× | 0 | 0.245 ± 0.004 | 0.00 ± 0.00 |
| 0,0 | 1.0× | 100 | 0.247 ± 0.005 | 0.00 ± 0.00 |
| 0,0 | 1.0× | 300 | 0.245 ± 0.000 | 0.00 ± 0.00 |
| 0,0 | 1.0× | 600 | 0.298 ± 0.128 | 0.01 ± 0.03 |
| 0,0 | 1.0× | 1200 | 0.995 ± 0.013 | 0.99 ± 0.03 |
| 0,0 | 1.0× | 2000 | 1.000 ± 0.000 | 1.00 ± 0.00 |
| 0,0 | 1.6× | 0 | 0.155 ± 0.001 | 0.00 ± 0.00 |
| 0,0 | 1.6× | 100 | 0.155 ± 0.000 | 0.00 ± 0.00 |
| 0,0 | 1.6× | 300 | 0.155 ± 0.000 | 0.00 ± 0.00 |
| 0,0 | 1.6× | 600 | 0.186 ± 0.071 | 0.00 ± 0.00 |
| 0,0 | 1.6× | 1200 | 0.965 ± 0.070 | 0.95 ± 0.11 |
| 0,0 | 1.6× | 2000 | 1.000 ± 0.000 | 1.00 ± 0.00 |
| 0,6 | 1.0× | 0 | 0.240 ± 0.011 | 0.00 ± 0.00 |
| 0,6 | 1.0× | 100 | 0.254 ± 0.012 | 0.00 ± 0.00 |
| 0,6 | 1.0× | 300 | 0.310 ± 0.000 | 0.00 ± 0.00 |
| 0,6 | 1.0× | 600 | 0.318 ± 0.058 | 0.00 ± 0.00 |
| 0,6 | 1.0× | 1200 | 1.000 ± 0.000 | 1.00 ± 0.00 |
| 0,6 | 1.0× | 2000 | 0.856 ± 0.399 | 0.81 ± 0.53 |
| 0,6 | 1.6× | 0 | 0.155 ± 0.004 | 0.00 ± 0.00 |
| 0,6 | 1.6× | 100 | 0.164 ± 0.015 | 0.00 ± 0.00 |
| 0,6 | 1.6× | 300 | 0.195 ± 0.000 | 0.00 ± 0.00 |
| 0,6 | 1.6× | 600 | 0.182 ± 0.031 | 0.00 ± 0.00 |
| 0,6 | 1.6× | 1200 | 1.000 ± 0.000 | 1.00 ± 0.00 |
| 0,6 | 1.6× | 2000 | 0.826 ± 0.465 | 0.79 ± 0.55 |

At episode 1,200, nominal survival was 99% at start (0,0) and 100% at (0,6). Shifted survival was 95% and 100%, respectively. At episode 2,000, start (0,0) remained at 100% in both conditions, but start (0,6) fell to 81% nominal and 79% shifted, with very wide intervals. This reflects one seed with a large regression; the other four start-(0,6) seeds retained full survival.

## Regression follow-up

Training seed 24004 from (0,6) was re-evaluated at checkpoints 1,200 and 2,000 over the same 200 evaluation-seed IDs in each metabolism condition. At 1,200, survival was 100% nominal and 99.5% at 1.6× metabolism. At 2,000, it fell to 2% nominal and 0% shifted; mean viability fell to 0.260 and 0.155. The 2,000-episode traces show repeated movement or consume actions at the top boundary without returning reliably to either resource. This confirms a learned-policy regression for that seed, rather than a simple shortage of evaluation episodes. The mechanism is not yet isolated; continued constant-step-size Q updates and the partially observed, noisy state representation are candidate causes, not established explanations.

## Qualification conclusion

The **world passes the feasibility check**: the hazard-avoiding scripted policy maintains viability and survives at both starts and both metabolism rates. The transition arithmetic checks also pass. The **learned controller does not yet pass robust-training qualification**: it can reach high survival by episode 1,200, but performance is seed-sensitive and can collapse by episode 2,000. Do not use a single selected checkpoint or this five-seed run as confirmatory evidence.

Next, freeze the transition semantics and preregister checkpoint selection using a separate validation set, then evaluate the selected checkpoint once on held-out seeds and starts. Compare a fixed 1,200-episode checkpoint against continued training to 2,000 to isolate training instability. Only proceed to causal interventions after this qualification step is stable.

The subsequent focused 20-seed checkpoint comparison found no clear average difference by paired 95% intervals, but showed start-dependent gains and losses across individual seeds. See [the focused checkpoint check](MEQ_focused_checkpoint_check.md) for the larger paired sample.

## Files

- [Episode-level diagnosis](MEQ_episode_level_diagnosis.md)
- [Per-seed checkpoint data](meq_full_budget_qualification_seed_results.csv)
- [Checkpoint summary JSON](meq_full_budget_qualification_summary.json)
- [Matched-seed regression episodes](meq_regression_deep_dive_episodes.csv)
- [Representative regression traces](meq_regression_deep_dive_traces.csv)
- [Full-budget diagnostic source](meq_full_budget_qualification.py)
- [Regression follow-up source](meq_regression_deep_dive.py)
