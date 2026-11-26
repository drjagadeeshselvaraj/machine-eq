import runpy, csv, json, math
from pathlib import Path
import numpy as np

BASE=runpy.run_path('outputs/meq_independent_reimplementation.py')
SIZE=BASE['SIZE']; FOOD=BASE['FOOD']; WATER=BASE['WATER']; HAZARDS=BASE['HAZARDS']; ACTIONS=BASE['ACTIONS']; STEPS=BASE['STEPS'];
train=BASE['train']; encode=BASE['encode']; observe=BASE['observe']; transition=BASE['transition'];

OUT=Path('outputs'); OUT.mkdir(exist_ok=True)
STARTS=[(0,0),(0,6)]
SEEDS=10; EPISODES=300; EVAL_PER_SEED=40
ACTION_NAMES=['up','right','down','left','consume']

def bfs_path(src,dst):
    todo=[src]; prev={src:None}; prev_action={}
    for p in todo:
        if p==dst: break
        for a,(dr,dc) in enumerate(ACTIONS[:4]):
            q=(max(0,min(6,p[0]+dr)),max(0,min(6,p[1]+dc)))
            if q in HAZARDS or q in prev: continue
            prev[q]=p; prev_action[q]=a; todo.append(q)
    if dst not in prev: raise ValueError(f'No safe route {src}->{dst}')
    path=[]; node=dst
    while node!=src: path.append((prev_action[node],node)); node=prev[node]
    return list(reversed(path))

def oracle_action(pos,target):
    if pos==target: return 4
    path=bfs_path(pos,target)
    return path[0][0]

def episode(policy,seed,start,rho=1.,q=None,log=False):
    rng=np.random.default_rng(seed); pos=start; x=np.ones(3,dtype=float)
    target=FOOD if start==(0,0) else WATER
    first={str(FOOD):None,str(WATER):None}; consumes={str(FOOD):0,str(WATER):0}
    action_counts=np.zeros(5,dtype=int); hazard_steps=0; viable=0; alive=0
    minx=np.ones(3); death_step=None; trace=[]
    for t in range(STEPS):
        if policy=='oracle':
            a=oracle_action(pos,target)
        elif policy=='random': a=int(rng.integers(5))
        elif policy=='learned':
            obs=observe(x,rng); s=encode(pos,obs); vals=q[s]
            a=int(rng.choice(np.flatnonzero(vals==vals.max())))
        else: raise ValueError(policy)
        oldpos=pos; action_counts[a]+=1
        pos2,x2,intake,hazard,dead=transition(pos,x.copy(),a,rho,rng)
        # Record first arrival and successful consumption (consumption indicated by intake).
        if pos2==FOOD and first[str(FOOD)] is None: first[str(FOOD)]=t+1
        if pos2==WATER and first[str(WATER)] is None: first[str(WATER)]=t+1
        if intake:
            label=str(pos2); consumes[label]+=1
            if policy=='oracle': target=WATER if target==FOOD else FOOD
        hazard_steps+=hazard
        alive += int(not dead)
        viable += int(not dead and bool(np.all(x2>=.25)))
        minx=np.minimum(minx,x2)
        if log:
            trace.append({'step':t+1,'row_before':oldpos[0],'col_before':oldpos[1],'action':ACTION_NAMES[a],
                          'row_after':pos2[0],'col_after':pos2[1],'energy':float(x2[0]),'hydration':float(x2[1]),'integrity':float(x2[2]),
                          'viable':int(not dead and bool(np.all(x2>=.25))),'hazard':hazard,'intake':intake,'dead':int(dead)})
        pos,x=pos2,x2
        if dead:
            death_step=t+1; break
    result={'policy':policy,'start':f'{start[0]},{start[1]}','rho':rho,'seed':seed,'viability_fraction':viable/STEPS,
            'alive_steps':alive,'survived_200':int(alive==STEPS),'death_step':death_step,'first_food_step':first[str(FOOD)],
            'first_water_step':first[str(WATER)],'food_intakes':consumes[str(FOOD)],'water_intakes':consumes[str(WATER)],
            'hazard_steps':hazard_steps,'min_energy':float(minx[0]),'min_hydration':float(minx[1]),'min_integrity':float(minx[2]),
            **{f'action_{ACTION_NAMES[i]}':int(action_counts[i]) for i in range(5)}}
    return result,trace

# Deterministic transition qualification checks from manuscript mechanics.
def transition_checks():
    checks=[]
    def check(name,passed,detail): checks.append({'check':name,'passed':bool(passed),'detail':detail})
    rng=np.random.default_rng(22)
    # decay away from resources/hazards: exact energy/hydration decrement
    _,x,*_=transition((0,6),np.ones(3),0,1.,rng)
    check('nominal_decay',np.allclose(x,[.988,.985,1.]),f'after one safe step x={x.tolist()} expected [.988,.985,1.0]')
    _,x,*_=transition((0,6),np.ones(3),0,1.6,np.random.default_rng(22))
    check('shifted_decay',np.allclose(x,[.9808,.976,1.]),f'after one safe 1.6x step x={x.tolist()} expected [.9808,.976,1.0]')
    _,x,intake,*_=transition(FOOD,np.ones(3),4,1.,np.random.default_rng(22))
    check('food_gain_and_decay',intake==1 and abs(x[0]-.988)<1e-9,f'food intake flag={intake}, poststep energy={x[0]:.3f}; expected .988 after +.7 cap then decay')
    _,x,intake,*_=transition(WATER,np.ones(3),4,1.,np.random.default_rng(22))
    check('water_gain_and_decay',intake==1 and abs(x[1]-.985)<1e-9,f'water intake flag={intake}, poststep hydration={x[1]:.3f}; expected .985 after +.7 cap then decay')
    _,x,_,hazard,*_=transition((2,2),np.ones(3),1,1.,np.random.default_rng(22)) # moves to (3,2), hazard
    check('hazard_damage',hazard==1 and abs(x[2]-.91)<1e-9,f'hazard flag={hazard}, integrity={x[2]:.3f}; expected .91 after damage and regen')
    # Energy falls below zero, clips to zero, and triggers death.
    _,x,_,_,dead=transition((0,6),np.array([.001,.3,.2]),0,1.,np.random.default_rng(22))
    check('death_and_clipping',dead and np.all(x>=0) and x[0]==0.,f'dead={dead}, poststep x={x.tolist()}')
    check('safe_oracle_paths',all(len(bfs_path(a,b))>0 for a in (FOOD,WATER) for b in (FOOD,WATER) if a!=b),'food and water are mutually reachable without hazard cells')
    return checks

checks=transition_checks()
# Scripted feasibility: 50 paired paths from each start, nominal and shifted.
all_rows=[]; traces=[]
for start in STARTS:
  for rho in (1.,1.6):
    for rep in range(50):
      r,tr=episode('oracle',900000+rep+(1000 if start==(0,6) else 0),start,rho,log=(rep==0))
      all_rows.append(r)
      if tr:
        for rr in tr: rr.update({'policy':'oracle','start':r['start'],'rho':rho,'rep':rep})
        traces.extend(tr)
# Random policy, same conditions, 100 per cell.
for start in STARTS:
  for rho in (1.,1.6):
    for rep in range(100): all_rows.append(episode('random',910000+rep+(1000 if start==(0,6) else 0),start,rho)[0])
# Learned-policy diagnostics: 10 paired seeds, full-factor arm only, at exact 300-episode budget.
for si,start in enumerate(STARTS):
  for seed_i in range(SEEDS):
    seed=10000+seed_i+si*1000
    Q=train(seed,start,'valence',True,episodes=EPISODES)
    for rho in (1.,1.6):
      for rep in range(EVAL_PER_SEED):
        this_seed=seed+50000+si*1000+rep+(100000 if rho>1 else 0)
        r,tr=episode('learned',this_seed,start,rho,q=Q,log=(rep==0))
        r['training_seed']=seed; r['eval_rep']=rep; all_rows.append(r)
        if tr:
          for rr in tr: rr.update({'policy':'learned','start':r['start'],'rho':rho,'training_seed':seed,'eval_rep':rep})
          traces.extend(tr)

# Add full machine-readable episodes and selected per-step traces.
fields=sorted({k for r in all_rows for k in r})
with (OUT/'meq_episode_diagnostic_episodes.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(all_rows)
trace_fields=sorted({k for r in traces for k in r})
with (OUT/'meq_episode_diagnostic_traces.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=trace_fields); w.writeheader(); w.writerows(traces)
with (OUT/'meq_simulator_qualification_checks.json').open('w',encoding='utf-8') as f: json.dump({'checks':checks,'all_passed':all(c['passed'] for c in checks)},f,indent=2)

# Summaries for each policy/start/rho.
summary=[]
for pol in ('oracle','random','learned'):
 for start in ('0,0','0,6'):
  for rho in (1.,1.6):
   rr=[r for r in all_rows if r['policy']==pol and r['start']==start and r['rho']==rho]
   if not rr: continue
   def avg(k): return float(np.mean([r[k] for r in rr]))
   summary.append({'policy':pol,'start':start,'rho':rho,'n':len(rr),'mean_viability':avg('viability_fraction'),
                   'survival_rate':avg('survived_200'),'mean_alive_steps':avg('alive_steps'),'mean_hazard_steps':avg('hazard_steps'),
                   'mean_food_intakes':avg('food_intakes'),'mean_water_intakes':avg('water_intakes'),
                   'mean_first_food_step':float(np.mean([r['first_food_step'] for r in rr if r['first_food_step'] is not None])) if any(r['first_food_step'] is not None for r in rr) else None,
                   'mean_first_water_step':float(np.mean([r['first_water_step'] for r in rr if r['first_water_step'] is not None])) if any(r['first_water_step'] is not None for r in rr) else None})
(OUT/'meq_simulator_qualification_summary.json').write_text(json.dumps({'checks':checks,'all_checks_passed':all(c['passed'] for c in checks),'summary':summary},indent=2),encoding='utf-8')

L=['# Episode level diagnosis and simulator qualification','',
'This diagnostic uses the independent implementation and its stated event-order assumptions. It asks whether the world admits safe behavior and where learned trajectories fail. It does not validate the original missing pilot code.','',
'## Transition checks','']
for c in checks: L.append(f"- **{'PASS' if c['passed'] else 'FAIL'} — {c['check']}:** {c['detail']}")
L+=['','The transition checks verify the implemented arithmetic and path reachability. They do not establish that each assumption matches the unavailable source implementation.','',
'## Episode outcomes','',
'| Policy | Start | Metabolism | Episodes | Mean viability | Survival rate | Mean steps alive | Mean hazard steps | Mean food intakes | Mean water intakes |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in summary:
 L.append(f"| {r['policy']} | {r['start']} | {r['rho']:.1f}× | {r['n']} | {r['mean_viability']:.3f} | {r['survival_rate']:.3f} | {r['mean_alive_steps']:.1f} | {r['mean_hazard_steps']:.2f} | {r['mean_food_intakes']:.2f} | {r['mean_water_intakes']:.2f} |")
L+=['','## Interpretation','',
'Scripted oracle cycles are a feasibility benchmark: if they preserve the viability set while the learned policy does not, the world itself is not intrinsically impossible under these mechanics. Random and learned episodes show whether the failure reflects exploration, state coverage, reward credit assignment, hazard avoidance, or resource-use behavior. The report and CSVs contain per-episode outcomes; the trace file contains one representative rollout per learned seed/start/metabolism and oracle condition.','',
'## Files','',
'- [Episode summary](meq_simulator_qualification_summary.json)','- [All episode outcomes](meq_episode_diagnostic_episodes.csv)','- [Representative step traces](meq_episode_diagnostic_traces.csv)','- [Transition-check results](meq_simulator_qualification_checks.json)','- [Diagnostic source](meq_episode_diagnosis.py)','']
(OUT/'MEQ_episode_level_diagnosis.md').write_text('\n'.join(L),encoding='utf-8')
print(json.dumps({'checks':checks,'summary':summary},indent=2))
