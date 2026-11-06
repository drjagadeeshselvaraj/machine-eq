import runpy,csv,json,math,time
from pathlib import Path
import numpy as np

B=runpy.run_path('outputs/meq_independent_reimplementation.py')
N_STATES=B['SIZE']*B['SIZE']*4**3; N_ACTIONS=5; STEPS=B['STEPS']
ACTIONS=B['ACTIONS']; FOOD=B['FOOD']; WATER=B['WATER']; HAZARDS=B['HAZARDS']
observe=B['observe']; encode=B['encode']; transition=B['transition']; run_episode=B['run_episode']
OUT=Path('outputs'); OUT.mkdir(exist_ok=True)
N_SEEDS=20; TRAIN_EPISODES=2000; CHECKPOINTS=(1200,2000); EVAL_EPISODES=50
STARTS=[(0,0),(0,6)]; RHOS=(1.0,1.6); ACTION_LABELS=('up','right','down','left','consume')


def evaluate(Q,seed,start,rho):
    rng=np.random.default_rng(seed); pos=start; x=np.ones(3,dtype=float)
    viable_steps=alive_steps=hazard_steps=food_intakes=water_intakes=0
    first_food=first_water=None; death_vars='none'; minx=np.ones(3,dtype=float)
    for t in range(STEPS):
        obs=observe(x,rng); state=encode(pos,obs); vals=Q[state]
        action=int(rng.choice(np.flatnonzero(vals==vals.max())))
        pos2,x2,intake,hazard,dead=transition(pos,x.copy(),action,rho,rng)
        if first_food is None and pos2==FOOD: first_food=t+1
        if first_water is None and pos2==WATER: first_water=t+1
        if intake:
            if pos2==FOOD: food_intakes+=1
            elif pos2==WATER: water_intakes+=1
        hazard_steps+=hazard
        alive_steps+=int(not dead)
        viable_steps+=int(not dead and bool(np.all(x2>=.25)))
        minx=np.minimum(minx,x2)
        if dead:
            death_vars=''.join(name for name,value in zip(('E','H','I'),x2) if value<=0) or 'unknown'
            break
        pos,x=pos2,x2
    return {'viability':viable_steps/STEPS,'survival':int(alive_steps==STEPS),'alive_steps':alive_steps,
            'death_vars':death_vars,'first_food_step':first_food,'first_water_step':first_water,
            'food_intakes':food_intakes,'water_intakes':water_intakes,'hazard_steps':hazard_steps,
            'min_energy':float(minx[0]),'min_hydration':float(minx[1]),'min_integrity':float(minx[2])}

def ci95(values):
    x=np.asarray(values,dtype=float); n=len(x); mean=float(x.mean()); sd=float(x.std(ddof=1))
    # t(19)=2.093 for this prespecified 20-seed comparison.
    return {'mean':mean,'half95':2.093*sd/math.sqrt(n),'n':n}

t0=time.time(); episodes=[]
for seed_i in range(N_SEEDS):
    training_seed=31000+seed_i
    for start_i,start in enumerate(STARTS):
        Q=np.zeros((N_STATES,N_ACTIONS),dtype=np.float32)
        rng=np.random.default_rng(training_seed)
        snapshots={}
        for ep in range(TRAIN_EPISODES):
            epsilon=1.-.95*min(ep,1200)/1200
            run_episode(Q,rng,start,reward='valence',sensing=True,epsilon=epsilon,learn=True)
            if ep+1 in CHECKPOINTS: snapshots[ep+1]=Q.copy()
        for checkpoint,Qc in snapshots.items():
            for rho_i,rho in enumerate(RHOS):
                for rep in range(EVAL_EPISODES):
                    # Same evaluation stream at each checkpoint for this seed/start/metabolism.
                    eval_seed=800000+seed_i*10000+start_i*1000+rho_i*500+rep
                    metrics=evaluate(Qc,eval_seed,start,rho)
                    episodes.append({'training_seed':training_seed,'start':f'{start[0]},{start[1]}','checkpoint':checkpoint,
                                     'rho':rho,'eval_rep':rep,'eval_seed':eval_seed,**metrics})
        print(f'completed seed {seed_i+1}/{N_SEEDS}, start {start}; elapsed {time.time()-t0:.1f}s',flush=True)

with (OUT/'meq_focused_checkpoint_episodes.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(episodes[0])); w.writeheader(); w.writerows(episodes)
summary=[]; paired=[]
for start in ('0,0','0,6'):
    for rho in RHOS:
        for checkpoint in CHECKPOINTS:
            seed_summaries=[]
            for seed in sorted({r['training_seed'] for r in episodes}):
                rr=[r for r in episodes if r['start']==start and r['rho']==rho and r['checkpoint']==checkpoint and r['training_seed']==seed]
                seed_summaries.append({'seed':seed,**{k:float(np.mean([r[k] for r in rr])) for k in ('viability','survival','alive_steps','food_intakes','water_intakes','hazard_steps')}})
            entry={'start':start,'rho':rho,'checkpoint':checkpoint,'n_training_seeds':N_SEEDS,
                   'viability':ci95([x['viability'] for x in seed_summaries]),
                   'survival':ci95([x['survival'] for x in seed_summaries]),
                   'alive_steps':ci95([x['alive_steps'] for x in seed_summaries]),
                   'food_intakes':ci95([x['food_intakes'] for x in seed_summaries]),
                   'water_intakes':ci95([x['water_intakes'] for x in seed_summaries]),
                   'hazard_steps':ci95([x['hazard_steps'] for x in seed_summaries]),
                   'seed_details':seed_summaries}
            summary.append(entry)
        for metric in ('viability','survival','alive_steps','food_intakes','water_intakes','hazard_steps'):
            deltas=[]
            for seed in sorted({r['training_seed'] for r in episodes}):
                by_ck={ck:[r for r in episodes if r['start']==start and r['rho']==rho and r['checkpoint']==ck and r['training_seed']==seed] for ck in CHECKPOINTS}
                m12=float(np.mean([r[metric] for r in by_ck[1200]])); m20=float(np.mean([r[metric] for r in by_ck[2000]]))
                deltas.append(m12-m20)
            paired.append({'start':start,'rho':rho,'contrast':'1200_minus_2000','outcome':metric,**ci95(deltas),
                           'n_seeds_checkpoint1200_survival_gt_0':sum(1 for seed in sorted({r['training_seed'] for r in episodes}) if np.mean([r['survival'] for r in episodes if r['start']==start and r['rho']==rho and r['checkpoint']==1200 and r['training_seed']==seed])>0),
                           'n_seeds_checkpoint2000_survival_gt_0':sum(1 for seed in sorted({r['training_seed'] for r in episodes}) if np.mean([r['survival'] for r in episodes if r['start']==start and r['rho']==rho and r['checkpoint']==2000 and r['training_seed']==seed])>0)})

payload={'design':{'n_training_seeds':N_SEEDS,'starts':[list(s) for s in STARTS],'checkpoints':list(CHECKPOINTS),'evaluation_episodes_per_seed_checkpoint_condition':EVAL_EPISODES,'metabolism':list(RHOS),'epsilon_schedule':'1.0 to 0.05 linearly over first 1200 training episodes; then 0.05'},'summary':summary,'paired_checkpoint_contrasts':paired}
(OUT/'meq_focused_checkpoint_summary.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')

lines=['# Focused checkpoint feasibility check','',
'This independent reimplementation compares the 1,200-episode checkpoint with continued training through episode 2,000. Twenty training seeds were paired across two safe starts. For each checkpoint, seed, start, and metabolic condition, the greedy policy was evaluated on the same 50 held-out episode seed IDs. Epsilon decayed from 1.0 to 0.05 over the first 1,200 episodes and remained 0.05 thereafter. This is a feasibility screen, not a reproduction of the unavailable original script or a preregistered confirmatory study.','',
'## Results','',
'Values are training-seed means ± 95% t half-intervals (n=20; t(19)=2.093).','',
'| Start | Metabolism | Checkpoint | Viability | Survival | Mean alive steps | Mean food intakes | Mean water intakes | Mean hazard steps |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for e in summary:
 def f(k):return f"{e[k]['mean']:.3f} ± {e[k]['half95']:.3f}"
 lines.append(f"| {e['start']} | {e['rho']:.1f}× | {e['checkpoint']} | {f('viability')} | {f('survival')} | {f('alive_steps')} | {f('food_intakes')} | {f('water_intakes')} | {f('hazard_steps')} |")
lines+=['','## Paired checkpoint comparison','',
'The paired contrast is checkpoint 1,200 minus checkpoint 2,000; positive survival or viability differences favor freezing at 1,200. Seed-level pairing controls for training randomness.','',
'| Start | Metabolism | Viability difference | Survival difference | Seeds with any survivors: 1,200 → 2,000 |','|---|---:|---:|---:|---:|']
for e in paired:
 if e['outcome']=='viability':
  s=next(x for x in paired if x['start']==e['start'] and x['rho']==e['rho'] and x['outcome']=='survival')
  lines.append(f"| {e['start']} | {e['rho']:.1f}× | {e['mean']:.3f} ± {e['half95']:.3f} | {s['mean']:.3f} ± {s['half95']:.3f} | {e['n_seeds_checkpoint1200_survival_gt_0']} → {e['n_seeds_checkpoint2000_survival_gt_0']} |")
lines+=['','## Interpretation','',
'This comparison estimates whether additional fixed-step Q updates after the first strong checkpoint preserve or degrade the learned policy. Interpret it together with per-seed results and death causes: a mean can obscure rare catastrophic regressions. A stable 1,200 checkpoint is only a candidate for validation-based checkpoint selection; it does not establish a universally correct stopping point. The next study should use a prespecified validation set for checkpoint selection and a separate untouched test set.','',
'## Files','',
'- [Per-episode outcomes](meq_focused_checkpoint_episodes.csv)','- [Summary and paired contrasts](meq_focused_checkpoint_summary.json)','- [Analysis source](meq_focused_checkpoint_check.py)','']
(OUT/'MEQ_focused_checkpoint_check.md').write_text('\n'.join(lines),encoding='utf-8')
print('Completed focused check. Wrote',len(episodes),'episode rows; elapsed %.1fs'%(time.time()-t0),flush=True)
