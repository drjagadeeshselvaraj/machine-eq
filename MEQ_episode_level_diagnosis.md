# Episode level diagnosis and simulator qualification

This diagnostic uses the independent implementation and its stated event-order assumptions. It asks whether the world admits safe behavior and where learned trajectories fail. It does not validate the original missing pilot code.

## Transition checks

- **PASS — nominal_decay:** after one safe step x=[0.988, 0.985, 1.0] expected [.988,.985,1.0]
- **PASS — shifted_decay:** after one safe 1.6x step x=[0.9808, 0.976, 1.0] expected [.9808,.976,1.0]
- **PASS — food_gain_and_decay:** food intake flag=1, poststep energy=0.988; expected .988 after +.7 cap then decay
- **PASS — water_gain_and_decay:** water intake flag=1, poststep hydration=0.985; expected .985 after +.7 cap then decay
- **PASS — hazard_damage:** hazard flag=1, integrity=0.910; expected .91 after damage and regen
- **PASS — death_and_clipping:** dead=True, poststep x=[0.0, 0.285, 0.2]
- **PASS — safe_oracle_paths:** food and water are mutually reachable without hazard cells

The transition checks verify the implemented arithmetic and path reachability. They do not establish that each assumption matches the unavailable source implementation.

## Episode outcomes

| Policy | Start | Metabolism | Episodes | Mean viability | Survival rate | Mean steps alive | Mean hazard steps | Mean food intakes | Mean water intakes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle | 0,0 | 1.0× | 50 | 1.000 | 1.000 | 200.0 | 0.00 | 11.00 | 11.00 |
| oracle | 0,0 | 1.6× | 50 | 1.000 | 1.000 | 200.0 | 0.00 | 11.00 | 11.00 |
| oracle | 0,6 | 1.0× | 50 | 1.000 | 1.000 | 200.0 | 0.00 | 11.00 | 11.00 |
| oracle | 0,6 | 1.6× | 50 | 1.000 | 1.000 | 200.0 | 0.00 | 11.00 | 11.00 |
| random | 0,0 | 1.0× | 100 | 0.240 | 0.000 | 64.9 | 5.65 | 0.71 | 0.05 |
| random | 0,0 | 1.6× | 100 | 0.154 | 0.000 | 40.9 | 3.42 | 0.52 | 0.01 |
| random | 0,6 | 1.0× | 100 | 0.242 | 0.000 | 65.8 | 6.00 | 0.18 | 0.21 |
| random | 0,6 | 1.6× | 100 | 0.156 | 0.000 | 41.6 | 3.24 | 0.10 | 0.13 |
| learned | 0,0 | 1.0× | 400 | 0.252 | 0.000 | 67.9 | 1.47 | 1.71 | 0.43 |
| learned | 0,0 | 1.6× | 400 | 0.159 | 0.000 | 42.1 | 0.98 | 1.49 | 0.32 |
| learned | 0,6 | 1.0× | 400 | 0.282 | 0.000 | 75.3 | 1.07 | 0.97 | 1.73 |
| learned | 0,6 | 1.6× | 400 | 0.174 | 0.000 | 46.3 | 0.79 | 0.60 | 1.25 |

In 400 learned nominal episodes per start, hydration reached zero first in 354/400 episodes from (0,0) and 229/400 from (0,6); energy reached zero first in 39/400 and 165/400, respectively. Integrity was first to reach zero in 7/400 and 6/400. Under 1.6× metabolism, hydration-first deaths were 355/400 and 227/400, while energy-first deaths were 40/400 and 171/400. Hazard exposure was comparatively low. The learned policy fails mainly because it does not reliably cycle between the resources.

## Interpretation

Scripted oracle cycles demonstrate feasibility: all 200 of 200 oracle episodes per start/metabolism condition survived with viability fraction 1.0 and zero hazard steps. The independent transition checks passed. The learned policy did not qualify: zero of 400 episodes survived in each start/metabolism condition. Its deaths were predominantly from hydration or energy depletion; it frequently failed to visit both resources before death. Random control performance was similarly nonviable. This points to inadequate learned resource cycling at 300 episodes, rather than an impossible world. A separate full-budget run reached high survival by episode 1,200, but one of five screening seeds from (0,6) regressed sharply by episode 2,000. A subsequent 20-seed paired check found mean survival between 0.914 and 1.000 at episode 2,000 across the four start/metabolism conditions; paired mean differences versus episode 1,200 had intervals spanning zero. Rare seed-level failures remained, including two of 20 seeds below 0.80 survival at the (0,0), 1.6× condition. The learned policy is broadly feasible but not yet robustly qualified against a preregistered tail-risk criterion, and these runs do not validate the original manuscript’s pilot. See the [full-budget qualification](MEQ_full_budget_qualification.md) and [20-seed paired check](MEQ_focused_checkpoint_check.md). The report and CSVs contain per-episode outcomes; the trace file contains one representative rollout per learned seed/start/metabolism and oracle condition.

## Files

- [Episode summary](meq_simulator_qualification_summary.json)
- [All episode outcomes](meq_episode_diagnostic_episodes.csv)
- [Representative step traces](meq_episode_diagnostic_traces.csv)
- [Transition-check results](meq_simulator_qualification_checks.json)
- [Diagnostic source](meq_episode_diagnosis.py)
- [Full-budget checkpoint results](MEQ_full_budget_qualification.md)
- [20-seed focused checkpoint comparison](MEQ_focused_checkpoint_check.md)
