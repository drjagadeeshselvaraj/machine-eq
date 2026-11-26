import runpy,csv,hashlib
from pathlib import Path
import numpy as np
B=runpy.run_path('outputs/meq_independent_reimplementation.py')
Q=np.zeros((B['SIZE']*B['SIZE']*4**3,5),dtype=np.float32)
seed=24004; start=(0,6); rng=np.random.default_rng(seed); snapshots={0:Q.copy()}
for ep in range(2000):
 eps=1.-.95*min(ep,1200)/1200
 B['run_episode'](Q,rng,start,reward='valence',sensing=True,epsilon=eps,learn=True)
 if ep+1 in (1200,2000): snapshots[ep+1]=Q.copy()

def logged(Q,evseed,rho):
 rng=np.random.default_rng(evseed); pos=start; x=np.ones(3); rows=[]; alive=0; viable=0; first={str(B['FOOD']):None,str(B['WATER']):None}; intake={'food':0,'water':0}
 for t in range(B['STEPS']):
  obs=B['observe'](x,rng); s=B['encode'](pos,obs); q=Q[s]; a=int(rng.choice(np.flatnonzero(q==q.max())))
  prev=pos; pos2,x2,got,hazard,dead=B['transition'](pos,x.copy(),a,rho,rng)
  alive+=int(not dead); viable+=int(not dead and np.all(x2>=.25))
  if pos2==B['FOOD'] and first[str(B['FOOD'])] is None: first[str(B['FOOD'])]=t+1
  if pos2==B['WATER'] and first[str(B['WATER'])] is None: first[str(B['WATER'])]=t+1
  if got: intake['food' if pos2==B['FOOD'] else 'water']+=1
  rows.append({'train_seed':seed,'checkpoint':1200 if Q is snapshots[1200] else 2000,'eval_seed':evseed,'rho':rho,'step':t+1,'row':pos2[0],'col':pos2[1],'action':B['ACTIONS'][a] if a!=4 else 'consume','energy':x2[0],'hydration':x2[1],'integrity':x2[2],'viable':int(not dead and np.all(x2>=.25)),'hazard':hazard,'intake':got,'dead':int(dead)})
  pos,x=pos2,x2
  if dead: break
 return {'checkpoint':1200 if Q is snapshots[1200] else 2000,'eval_seed':evseed,'rho':rho,'viability':viable/200,'survival':int(alive==200),'alive_steps':alive,'food_intakes':intake['food'],'water_intakes':intake['water'],'first_food':first[str(B['FOOD'])],'first_water':first[str(B['WATER'])]},rows

results=[]; trace=[]
for rho in (1.,1.6):
 for ck,Qc in snapshots.items():
  for j in range(200):
   r,tr=logged(Qc,500000+j,rho); r['checkpoint']=ck; results.append(r)
   if j<5:
    for row in tr: trace.append(row)
with open('outputs/meq_regression_deep_dive_episodes.csv','w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=list(results[0])); w.writeheader(); w.writerows(results)
with open('outputs/meq_regression_deep_dive_traces.csv','w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=list(trace[0])); w.writeheader(); w.writerows(trace)
for ck in (1200,2000):
 for rho in (1.,1.6):
  rr=[r for r in results if r['checkpoint']==ck and r['rho']==rho]
  print(ck,rho,'n',len(rr),'survival',np.mean([r['survival'] for r in rr]),'viability',np.mean([r['viability'] for r in rr]),'food/water',np.mean([r['food_intakes'] for r in rr]),np.mean([r['water_intakes'] for r in rr]))
