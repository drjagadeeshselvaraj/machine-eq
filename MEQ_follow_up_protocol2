# MEQ follow-up protocol and independent reimplementation

## Status and scope

This package is a follow-up research plan plus an **independent reimplementation** of the environment described in the manuscript. The manuscript’s cited `meq_pilot.py` source was not available in the supplied files. The simulation output therefore does not reproduce, verify, or replace the manuscript’s reported pilot values. The initial analysis is exploratory and was run after the manuscript and its pilot findings had already been inspected. The plan below is a draft for prospective registration before any confirmatory simulation or robot data are collected; it has not been registered externally.

The public RoboBATT analysis in the manuscript remains a descriptive battery-telemetry analysis. It is not an evaluation of MEQ, a controller intervention, or evidence of affective regulation.

## Independent simulation implementation

The implementation uses the manuscript’s 7 × 7 grid, resource locations, five-cell central hazard, three internal variables, Q-learning state/action dimensions, sensor bin edges, reward descriptions, and nominal/1.6× metabolic conditions. The matched 2 × 2 factorial crosses interoceptive observations (noisy sensor vs fixed null bins) with reward (drive reduction vs task reward). All cells use the same 3,136-state table capacity, five actions, learning rate, discount, exploration schedule, training budget, and evaluation protocol. Each seed trains all four cells. The completed preliminary run used 10 paired seeds, 300 episodes per cell, 40 greedy evaluation episodes per seed, and two safe corner starts, (0,0) and (0,6). No evaluation episode survived to the 200-step horizon. This indicates that the selected short budget did not produce competent policies; the measured differences are therefore a design check rather than evidence for or against MEQ. For the completed short run, epsilon was rescaled to reach 0.05 at episode 300; this differs from the manuscript’s 1,200-episode decay. The short budget and altered exploration schedule are for debugging whether the design yields useful contrasts; they are not a convergence claim. The confirmatory protocol restores the manuscript’s linear decay over the first 1,200 episodes.

Two test-time interventions are applied to the trained sensing-plus-valence policy: (1) the manuscript’s fixed-bin clamp (energy bin 3, hydration bin 3, integrity bin 4), and (2) an independent donor shuffle that samples joint sensor-bin tuples from an empirical intact-policy sensor pool. The donor shuffle retains the empirical joint marginal over the three readings while breaking the alignment between the current internal state and the sensor tuple. Each intervention is assessed at nominal and 1.6× metabolism without retraining. Outcomes are viability fraction (steps with all three variables at or above θ = 0.25, divided by 200) and episode survival to step 200. The nominal per-seed gate is reported as C = (P_intact − P_intervention) / P_intact when P_intact > 0. Paired seed values and 95% t intervals are reported; with 10 seeds, uncertainty remains substantial.

### Explicit implementation assumptions

The manuscript does not specify the initial location, boundary behavior, the exact timing of sensing, movement, consumption, decay, hazard damage, regeneration, clipping, and death, or the training-evaluation transition details. This implementation makes those choices explicit in its source: safe corner starts; bounded movement at edges; observation before action; intake or movement, then resource decay, hazard damage, conditional integrity regeneration, clipping, then death check. The internal-state variables are initialized to one. Noise is Gaussian with σ = 0.05 and readings are clipped before discretization. These choices are not asserted to match the missing source. The two-start comparison is a narrow robustness check, not an exhaustive layout sensitivity analysis.

The original manuscript’s baseline used timers and a different reward. The new factorial instead matches representational capacity and directly separates the presence of sensed internal variables from reward type. It consequently answers a controlled question about this reimplementation, not the original affective-vs-clocked comparison.

## Prospective confirmatory simulation preregistration draft

### Hypotheses

1. With reward held fixed, sensing the three internal variables improves viability and survival over fixed null readings.
2. With sensing held fixed, drive-reduction reward improves viability over task reward, or there is an interaction between reward and sensing.
3. Breaking the alignment between current internal state and sensor readings reduces viability more than intact sensing. The donor-shuffle intervention is the primary causal test; the fixed clamp is a secondary intervention because it can create out-of-distribution inputs.
4. The pattern persists under the prespecified 1.6× metabolism shift and across prespecified safe start cells.

### Confirmatory design and analysis

Before running the confirmatory study, freeze the source, environment parameters, package versions, seed list, and analysis script; deposit a time-stamped protocol and code snapshot in a public registry. Use 30 paired training seeds, 2,000 training episodes per cell, and 200 greedy evaluation episodes per seed and condition. Predefine at least four safe starts and two unseen resource/hazard layouts. Do not tune parameters after unblinding cell labels. Train the 2 × 2 factorial with common seed IDs and identical training budgets. Evaluate nominal metabolism and a 1.6× shift without retraining. The primary outcome is per-seed viability fraction; survival and reward are secondary. Report arm means, paired contrasts, 95% intervals, per-seed data, and all prespecified interactions. Use a hierarchical model with seed as the pairing/blocking unit as a sensitivity analysis; do not treat episodes as independent training replicates.

For the gate, compare intact sensing with donor-shuffled readings within the same trained sensing-plus-valence policy. Report paired absolute differences in viability and survival as primary intervention estimates and C as a secondary normalized summary. Report fixed clamp separately. Also include a retrained no-sensor policy as a learning-adaptation comparison; do not call that retrained contrast a test-time causal gate. Include sensor-noise and donor-pool sensitivity analyses, and report failures, exclusions, and all seeds.

### Checkpoint stability requirement

The subsequent full-budget qualification found a sharp regression in one seed after performance had become strong by episode 1,200. The fixed-final-checkpoint design above should therefore be amended before registration: retain prespecified checkpoints including 1,200 and 2,000 episodes; select any deployed checkpoint only using a separate validation set and a rule frozen before training; and report the 2,000-episode endpoint alongside the selected checkpoint. Keep confirmatory test episodes and held-out starts/layouts untouched until the selection rule is applied. Do not choose a checkpoint by looking at the final test results. The qualification report contains the screening evidence and matched-seed regression follow-up. A subsequent 20-seed paired check showed no clear mean checkpoint difference by 95% intervals, but individual seeds improved or regressed depending on start and metabolism. The next registration should prespecify both a validation-based checkpoint rule and per-seed regression reporting; neither checkpoint is established as universally superior.

### Decision criteria

Evidence for useful sensor dependence requires a positive paired intact-minus-donor-shuffle viability contrast with a 95% interval excluding zero, the same direction for survival, and consistency across at least three of four starts and both layouts. The reward main effect and sensing-by-reward interaction are reported whether or not they meet this criterion. These criteria establish only policy dependence on the measured channel under this simulator; they do not validate affect, emotion, consciousness, or a general MEQ construct.

## Prospective real-system study protocol

### Access prerequisite and study system

A real-system study requires authorized controller access to the target robot, a documented API or firmware build, a safe test area, and permission to intercept or replace the candidate internal channel. The current public RoboBATT files do not provide controller state, policy reward, intervention access, calibrated independent ground truth, or a BMS response timestamp. They cannot support this causal experiment. If the operator-identified AGIBOT G2 Pro is selected, confirm the exact unit, firmware, sensor stack, payload, battery, and control interface before registration. No robot evaluation has yet been performed.

### Measures and protocol

Register one primary regulated variable with an independent reference instrument, operating limits, minimum/maximum safe range, sensor precision, and event timestamps. Primary outcome: fraction of control intervals inside the preregistered safe viability set. Secondary outcomes: time-to-limit, intervention recovery time, task completion, energy/current, sensor-reference error, and safety-stop count. Log raw reference measurements, candidate internal estimates, controller observations/actions, reward or objective signals if accessible, intervention assignments, and synchronized timestamps.

Within each robot and randomized block, compare: intact channel; a preregistered in-distribution donor channel sampled from matched states and task phases; fixed clamp; and a separately trained policy without the channel. The donor intervention is primary for the causal gate; fixed clamp and retraining comparisons are secondary. Randomize condition order within safe operating blocks, use matched task scripts and workload, include washout/reset criteria, and blind outcome analysis to condition labels. Do not deliberately approach hardware damage thresholds; choose conservative limits approved by the responsible operator and stop on any safety limit, sensor fault, unexpected motion, or human intervention.

Determine sample size from a pilot variance estimate collected before the confirmatory experiment; preregister the minimum meaningful paired change, stopping rules, missing-data handling, and analysis model before condition labels are revealed. Analyze paired block-level outcomes, with robot/unit and session as repeated-measures factors. Report all conditions, exclusions, interruptions, calibration records, and confidence intervals. Compute a controller-specific gate only when the intervention actually changes the accessible internal channel while preserving other inputs and logging confirms delivery.

### Claims boundary

Simulation supports claims only about the specified simulated policy and intervention. Public telemetry supports descriptive associations only. Real-system causal claims require controller-level intervention and independent measurements. None of these behavioral or control results alone establishes subjective experience.

## Files and reproducibility

- `outputs/meq_independent_reimplementation.py`: independent simulation source with implementation assumptions in comments; run from the project root with `python outputs/meq_independent_reimplementation.py`.
- `outputs/meq_independent_factorial_seed_results.csv`: paired per-seed factorial outcomes.
- `outputs/meq_independent_intervention_seed_results.csv`: per-seed intervention outcomes.
- `outputs/meq_independent_summary.json`: means and 95% t intervals.
- `outputs/MEQ_independent_simulation_results.md`: concise analysis and interpretation.
- `outputs/MEQ_episode_level_diagnosis.md`: transition checks and episode-level failure diagnosis.
- `outputs/MEQ_full_budget_qualification.md`: 2,000-episode checkpoint and regression analysis.
- `outputs/MEQ_focused_checkpoint_check.md`: 20-seed paired checkpoint feasibility check.

The independent simulation source SHA-256 is `cc109e8aa640404251b8116708390bdaac8a70a7dc4b9c40109d925ae4789f98`. The report should be regenerated from the source and the CSV files; no result should be copied into the manuscript as an original-pilot result.
