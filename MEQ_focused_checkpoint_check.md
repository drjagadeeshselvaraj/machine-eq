# Focused checkpoint feasibility check

This independent reimplementation compares the 1,200-episode checkpoint with continued training through episode 2,000. Twenty training seeds were paired across two safe starts. For each checkpoint, seed, start, and metabolic condition, the greedy policy was evaluated on the same 50 held-out episode seed IDs. Epsilon decayed from 1.0 to 0.05 over the first 1,200 episodes and remained 0.05 thereafter. This is a feasibility screen, not a reproduction of the unavailable original script or a preregistered confirmatory study.

## Results

Values are training-seed means ± 95% t half-intervals (n=20; t(19)=2.093).

| Start | Metabolism | Checkpoint | Viability | Survival | Mean alive steps | Mean food intakes | Mean water intakes | Mean hazard steps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0,0 | 1.0× | 1200 | 0.999 ± 0.002 | 0.997 ± 0.003 | 199.791 ± 0.298 | 9.087 ± 3.157 | 7.806 ± 1.032 | 1.837 ± 1.796 |
| 0,0 | 1.0× | 2000 | 0.982 ± 0.035 | 0.971 ± 0.051 | 196.934 ± 5.773 | 7.907 ± 2.624 | 7.388 ± 0.230 | 1.122 ± 1.203 |
| 0,0 | 1.6× | 1200 | 0.981 ± 0.013 | 0.972 ± 0.016 | 196.520 ± 2.477 | 10.871 ± 2.114 | 10.053 ± 0.507 | 1.713 ± 1.669 |
| 0,0 | 1.6× | 2000 | 0.946 ± 0.077 | 0.914 ± 0.120 | 190.317 ± 13.888 | 9.609 ± 1.404 | 9.552 ± 0.662 | 1.504 ± 1.487 |
| 0,6 | 1.0× | 1200 | 0.981 ± 0.036 | 0.961 ± 0.077 | 197.001 ± 5.735 | 13.498 ± 7.416 | 11.832 ± 8.709 | 0.861 ± 0.543 |
| 0,6 | 1.0× | 2000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 200.000 ± 0.000 | 7.625 ± 1.179 | 8.317 ± 1.999 | 1.334 ± 1.389 |
| 0,6 | 1.6× | 1200 | 0.939 ± 0.069 | 0.902 ± 0.103 | 189.089 ± 12.743 | 12.559 ± 5.504 | 13.627 ± 7.768 | 0.880 ± 0.726 |
| 0,6 | 1.6× | 2000 | 0.999 ± 0.002 | 0.998 ± 0.003 | 199.806 ± 0.292 | 10.084 ± 0.734 | 10.777 ± 1.759 | 1.572 ± 1.599 |

## Paired checkpoint comparison

The paired contrast is checkpoint 1,200 minus checkpoint 2,000; positive survival or viability differences favor freezing at 1,200. Seed-level pairing controls for training randomness.

| Start | Metabolism | Viability difference | Survival difference | Seeds with ≥1 survivor in 50: 1,200 → 2,000 |
|---|---:|---:|---:|---:|
| 0,0 | 1.0× | 0.017 ± 0.035 | 0.026 ± 0.051 | 20 → 20 |
| 0,0 | 1.6× | 0.035 ± 0.080 | 0.058 ± 0.124 | 20 → 20 |
| 0,6 | 1.0× | -0.019 ± 0.036 | -0.039 ± 0.077 | 20 → 20 |
| 0,6 | 1.6× | -0.060 ± 0.070 | -0.096 ± 0.103 | 19 → 20 |

The paired 95% intervals for the mean checkpoint differences include zero in every start/metabolism condition. Per-seed ranges still show substantial heterogeneity: at start (0,0), per-seed survival at 2,000 episodes ranged from 0.52 to 1.00 nominally and 0.10 to 1.00 under shifted metabolism; one seed was below 0.80 nominally and two were below 0.80 under the shift. At start (0,6), every seed had at least one survivor at both checkpoints, and all 2,000-episode seeds survived all nominal evaluation episodes; the weakest 1,200-episode seed improved after continued training. Thus the earlier collapse is not a general average effect of continuing training, while seed-specific regressions and recoveries remain visible. The earlier stress seed 24004 is documented separately in the full-budget report and is not included in this new 20-seed sample.

## Interpretation

This comparison estimates whether additional fixed-step Q updates after the first strong checkpoint preserve or degrade the learned policy. Interpret it together with per-seed results and death causes: a mean can obscure rare catastrophic regressions. A stable 1,200 checkpoint is only a candidate for validation-based checkpoint selection; it does not establish a universally correct stopping point. The next study should use a prespecified validation set for checkpoint selection and a separate untouched test set. Define the acceptable survival and regression criteria before training; the current comparison is exploratory and does not set those thresholds.

## Files

- [Per-episode outcomes](meq_focused_checkpoint_episodes.csv)
- [Summary and paired contrasts](meq_focused_checkpoint_summary.json)
- [Analysis source](meq_focused_checkpoint_check.py)
