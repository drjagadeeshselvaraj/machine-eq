import runpy,csv,json,math
from pathlib import Path
import numpy as np
B=runpy.run_path('outputs/meq_independent_reimplementation.py')
run_episode=B['run_episode']; eval_many=B['eval_many']; SIZE=B['SIZE']; EP=2000; SEEDS=5; CHECKS=[0,100,300,600,1200,2000]
OUT=Path('outputs'); results=[]
for si,start in enumerate([(0,0),(0,6)]):
 for j in range(SEEDS):
  seed=23000+j+si*1000
  Q=np.zeros((SIZE*SIZE*4**3,5),dtype=np.float32)
  if 0 in CHECKS:
   for rho in (1.,1.6):
    v,s=eval_many(Q,seed+50000,start,'valence',True,rho=rho,n=20)
    results.append({'start':f'{start[0]},{start[1]}','seed':seed,'checkpoint':0,'rho':rho,'viability':v,'survival':s})
  rng=np.random.default_rng(seed)
  for ep in range(EP):
   eps=1.-.95*min(ep,1200)/1200
   run_episode(Q,rng,start,reward='valence',sensing=True,epsilon=eps,learn=True)
   done=ep+1
   if done in CHECKS:
    for rho in (1.,1.6):
     v,s=eval_many(Q,seed+50000+done*7,start,'valence',True,rho=rho,n=20)
     results.append({'start':f'{start[0]},{start[1]}','seed':seed,'checkpoint':done,'rho':rho,'viability':v,'survival':s})
  print('trained',start,seed,flush=True)
with (OUT/'meq_full_budget_qualification_seed_results.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=list(results[0])); w.writeheader(); w.writerows(results)
def t_summary(vals):
 n=len(vals); m=float(np.mean(vals)); sd=float(np.std(vals,ddof=1)); crit=2.776 if n==5 else 2.262
 return {'mean':m,'half95':crit*sd/math.sqrt(n),'n':n}
summary=[]
for start in ('0,0','0,6'):
 for rho in (1.,1.6):
  for ck in CHECKS:
   rr=[r for r in results if r['start']==start and r['rho']==rho and r['checkpoint']==ck]
   if rr:
    summary.append({'start':start,'rho':rho,'checkpoint':ck,'viability':t_summary([r['viability'] for r in rr]),'survival':t_summary([r['survival'] for r in rr])})
(OUT/'meq_full_budget_qualification_summary.json').write_text(json.dumps({'episodes':EP,'seeds_per_start':SEEDS,'eval_episodes_per_seed_checkpoint':20,'epsilon_decay':'linear 1.0 to 0.05 over first 1200 episodes, then 0.05','summary':summary},indent=2),encoding='utf-8')
lines=['# Full budget simulator qualification','',
'This independent reimplementation was trained for the manuscript’s stated 2,000 episodes with epsilon decaying from 1.0 to 0.05 over the first 1,200 episodes. Five paired seeds per start were evaluated at each checkpoint using 20 greedy episodes per seed and metabolism condition. The run uses the explicitly documented transition assumptions; it is not a reproduction of the unavailable original pilot source.','',
'## Results','',
'Values are seed-level means ± 95% t half-intervals (five seeds; t(4) critical value 2.776).','',
'| Start | Metabolism | Episode checkpoint | Viability fraction | Survival |','|---|---:|---:|---:|---:|']
for x in summary:
 lines.append(f"| {x['start']} | {x['rho']:.1f}× | {x['checkpoint']} | {x['viability']['mean']:.3f} ± {x['viability']['half95']:.3f} | {x['survival']['mean']:.3f} ± {x['survival']['half95']:.3f} |")
lines+=['','## Qualification rule','',
'The environment passes a basic feasibility check because an independently scripted safe policy survives. The learned controller passes qualification only if the full-budget trained agent shows stable improvement and nonzero survival on the held-out evaluation episodes at both starts. Interpret five seeds as a screening result, not a confirmatory estimate.','',
'## Files','',
'- [Per-seed checkpoint evaluations](meq_full_budget_qualification_seed_results.csv)','- [Summary JSON](meq_full_budget_qualification_summary.json)','- [Full-budget diagnostic source](meq_full_budget_qualification.py)','']
(OUT/'MEQ_full_budget_qualification.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(summary,indent=2))
