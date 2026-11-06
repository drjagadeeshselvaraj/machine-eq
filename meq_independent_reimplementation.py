import numpy as np, json, csv, time
from pathlib import Path

# Independent reimplementation from manuscript prose; not the missing meq_pilot.py.
SIZE=7; FOOD=(1,1); WATER=(5,5)
HAZARDS={(2,3),(3,2),(3,3),(3,4),(4,3)}
ACTIONS=((0,-1),(1,0),(0,1),(-1,0),None)  # up, right, down, left, consume
ALPHA=.1; GAMMA=.95; EPISODES=300; EVAL_EPISODES=40; STEPS=200
SEEDS=10; SIGMA=.05; STARTS=[(0,0),(0,6)]


def bins(x):
    return int(np.digitize(x, [.25,.50,.75], right=False))

def encode(pos, obs):
    idx=pos[0]*SIZE+pos[1]
    for b in obs: idx=idx*4+int(b)
    return idx

def transition(pos, x, action, rho, rng):
    """One step. Assumption: act/consume, decay, hazard, then regen; clip; then death."""
    nr,nc=pos; intake=0; hazard=0
    delta=ACTIONS[action]
    if delta is None:
        if pos==FOOD:
            x[0]=min(1.,x[0]+.7); intake=1
        elif pos==WATER:
            x[1]=min(1.,x[1]+.7); intake=1
    else:
        nr=max(0,min(SIZE-1,nr+delta[0])); nc=max(0,min(SIZE-1,nc+delta[1])); pos=(nr,nc)
    # Explicit event order needed because manuscript does not specify it.
    x[0]-=.012*rho; x[1]-=.015*rho
    if pos in HAZARDS:
        x[2]-=.10; hazard=1
    if x[0]>.3 and x[1]>.3: x[2]+=.01
    x[:]=np.clip(x,0,1)
    dead=bool(np.any(x<=0))
    return pos,x,intake,hazard,dead

def observe(x,rng):
    return tuple(bins(float(np.clip(v+rng.normal(0,SIGMA),0,1))) for v in x)

def run_episode(Q, rng, start, rho=1., reward='valence', sensing=True, epsilon=0., intervention='intact', donor_pool=None, learn=True):
    pos=start; x=np.ones(3,dtype=float); alive_steps=0; viable_steps=0; total_r=0.
    for t in range(STEPS):
        obs=observe(x,rng) if sensing else (2,2,3)
        if intervention=='clamp': obs=(2,2,3)
        elif intervention=='donor_shuffle': obs=donor_pool[int(rng.integers(len(donor_pool)))]
        s=encode(pos,obs)
        if rng.random()<epsilon: a=int(rng.integers(5))
        else:
            vals=Q[s]; a=int(rng.choice(np.flatnonzero(vals==vals.max())))
        olddrive=float(np.sum((1.-x)**2))
        pos2,x2,intake,hazard,dead=transition(pos,x.copy(),a,rho,rng)
        if reward=='valence': r=olddrive-float(np.sum((1.-x2)**2))-(5. if dead else 0.)
        else: r=2.*intake-.5*hazard-(5. if dead else 0.)
        alive_steps += int(not dead)
        viable_steps += int(not dead and bool(np.all(x2 >= .25)))
        total_r += r
        if dead:
            if learn: Q[s,a] += ALPHA*(r - Q[s,a])
            break
        obs2=observe(x2,rng) if sensing else (2,2,3)
        if intervention=='clamp': obs2=(2,2,3)
        elif intervention=='donor_shuffle': obs2=donor_pool[int(rng.integers(len(donor_pool)))]
        s2=encode(pos2,obs2)
        # Q update uses fixed epsilon-greedy training policy; terminal target is zero.
        if learn: Q[s,a] += ALPHA*(r + (0. if dead else GAMMA*np.max(Q[s2])) - Q[s,a])
        pos,x=pos2,x2
    return viable_steps/STEPS, alive_steps, total_r

def train(seed,start,reward,sensing,episodes=EPISODES):
    rng=np.random.default_rng(seed)
    Q=np.zeros((SIZE*SIZE*4**3,5),dtype=np.float32)
    # Exploratory run rescales the linear epsilon decay to this short budget.
    # The prospective confirmatory run should restore the manuscript's 1,200-episode decay.
    for ep in range(episodes):
        eps=1.-.95*min(ep,episodes)/episodes
        run_episode(Q,rng,start,reward=reward,sensing=sensing,epsilon=eps)
    return Q

def eval_many(Q,seed,start,reward,sensing,rho=1.,intervention='intact',donor_pool=None,n=EVAL_EPISODES):
    rng=np.random.default_rng(seed)
    vals=[]; survives=[]
    for _ in range(n):
        v,steps,_=run_episode(Q,rng,start,rho,reward,sensing,0.,intervention,donor_pool,learn=False)
        vals.append(v); survives.append(float(steps==STEPS))
    return float(np.mean(vals)),float(np.mean(survives))

def make_donor_pool(Q,seed,start):
    rng=np.random.default_rng(seed+7919); pool=[]
    # Empirical intact-policy sensor distribution, from separate trajectories.
    for _ in range(40):
        pos=start; x=np.ones(3)
        for t in range(STEPS):
            pool.append(observe(x,rng))
            s=encode(pos,pool[-1]); q=Q[s]; a=int(np.flatnonzero(q==q.max())[0])
            pos,x,_,_,dead=transition(pos,x.copy(),a,1.,rng)
            if dead: break
    return pool

def ci_t95(x):
    # Small-n t critical value; use scipy when available.
    a=np.asarray(x,float); n=len(a); sd=float(a.std(ddof=1))
    try:
        from scipy.stats import t
        crit=float(t.ppf(.975,n-1))
    except Exception: crit=2.093 if n==20 else 2.262
    return float(a.mean()),float(crit*sd/np.sqrt(n))

def main():
    out=Path('outputs'); out.mkdir(exist_ok=True)
    rows=[]; seed_rows=[]; t0=time.time()
    # Primary factorial, 4 safe corner starts, matched policies, same seed set.
    for start_i,start in enumerate(STARTS):
      for si in range(SEEDS):
        seed=10000+si
        for sensing in (True,False):
          for reward in ('valence','task'):
            Q=train(seed+start_i*1000,start,reward,sensing)
            v,s=eval_many(Q,seed+50000+start_i*1000,start,reward,sensing)
            vs,ss=eval_many(Q,seed+60000+start_i*1000,start,reward,sensing,rho=1.6)
            seed_rows.append(dict(start=f'{start[0]},{start[1]}',seed=seed,sensing=int(sensing),reward=reward,nominal_viability=v,nominal_survival=s,shift_viability=vs,shift_survival=ss))
            # Causal interventions on the valence + sensing arm only.
            if sensing and reward=='valence':
              donor=make_donor_pool(Q,seed+70000+start_i*1000,start)
              for condition,inter in [('intact','intact'),('fixed_clamp','clamp'),('donor_shuffle','donor_shuffle')]:
                iv,isurv=eval_many(Q,seed+80000+start_i*1000,start,reward,sensing,intervention=inter,donor_pool=donor)
                ivs,iss=eval_many(Q,seed+90000+start_i*1000,start,reward,sensing,rho=1.6,intervention=inter,donor_pool=donor)
                rows.append(dict(start=f'{start[0]},{start[1]}',seed=seed,condition=condition,nominal_viability=iv,nominal_survival=isurv,shift_viability=ivs,shift_survival=iss))
      print(f'finished start {start} in {time.time()-t0:.1f}s',flush=True)
    with (out/'meq_independent_factorial_seed_results.csv').open('w',newline='',encoding='utf-8') as f:
      w=csv.DictWriter(f,fieldnames=list(seed_rows[0])); w.writeheader(); w.writerows(seed_rows)
    with (out/'meq_independent_intervention_seed_results.csv').open('w',newline='',encoding='utf-8') as f:
      w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summaries=[]
    for start in STARTS:
      key=f'{start[0]},{start[1]}'
      for sensing in (1,0):
       for reward in ('valence','task'):
        ss=[r for r in seed_rows if r['start']==key and r['sensing']==sensing and r['reward']==reward]
        for outcome in ('nominal_viability','nominal_survival','shift_viability','shift_survival'):
         m,h=ci_t95([r[outcome] for r in ss]); summaries.append(dict(start=key,arm=f"sensing_{sensing}_{reward}",outcome=outcome,mean=m,half95=h,n=len(ss)))
      ints=[r for r in rows if r['start']==key]
      for condition in ('intact','fixed_clamp','donor_shuffle'):
       ss=[r for r in ints if r['condition']==condition]
       for outcome in ('nominal_viability','nominal_survival','shift_viability','shift_survival'):
        m,h=ci_t95([r[outcome] for r in ss]); summaries.append(dict(start=key,arm=condition,outcome=outcome,mean=m,half95=h,n=len(ss)))
      # Per-seed gate C=(Pint-Pabl)/Pint for nominal viability; paired by seed.
      a={r['seed']:r['nominal_viability'] for r in ints if r['condition']=='intact'}
      for condition in ('fixed_clamp','donor_shuffle'):
       b={r['seed']:r['nominal_viability'] for r in ints if r['condition']==condition}
       gates=[(a[k]-b[k])/a[k] if a[k]>0 else 0. for k in a]
       m,h=ci_t95(gates); summaries.append(dict(start=key,arm=f'gate_{condition}',outcome='C_per_seed',mean=m,half95=h,n=len(gates)))
    (out/'meq_independent_summary.json').write_text(json.dumps({'status':'independent reimplementation; exploratory, not preregistered; not a reproduction of the missing script','episodes_per_seed':EPISODES,'seeds':SEEDS,'evaluation_episodes_per_seed':EVAL_EPISODES,'starts':[list(x) for x in STARTS],'summary':summaries},indent=2),encoding='utf-8')
    print('Wrote results; elapsed %.1fs'%(time.time()-t0))
    for s in summaries:
      if s['outcome'] in ('nominal_viability','shift_viability','C_per_seed'):
       print(s)
if __name__=='__main__': main()
