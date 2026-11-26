"""
MEQ pilot simulation (reference implementation for the book
"Machine Emotional Quotient (MEQ): From Artificial Intelligence to Artificial Feeling").

A tabular, vectorised foraging world with three internal variables
(energy e, hydration h, integrity d). Agents:
  - affective : observes a noisy interoceptive reading, rewarded by drive reduction
  - clocked   : non-affective controller with exteroceptive timers, rewarded by intake
  - random    : uniform random actions (reference for the viability baseline V0)
  - faker     : clocked policy plus a report channel that is never used
Outputs results.json and per-episode traces used for the figures.
"""
import json
import sys
import time
import numpy as np

# ------------------------------------------------------------------ world
N, A, T = 7, 5, 200
FOOD, WATER = (1, 1), (5, 5)
HAZ = np.zeros((N, N), bool)
for rc in [(3, 2), (3, 3), (3, 4), (2, 3), (4, 3)]:
    HAZ[rc] = True
DR = np.array([-1, 1, 0, 0, 0])
DC = np.array([0, 0, -1, 1, 0])
E_DEC, H_DEC, GAIN = 0.012, 0.015, 0.7
HAZ_DMG, REGEN, THETA = 0.10, 0.01, 0.25
EDGES = np.array([0.25, 0.5, 0.75])
CENTERS = np.array([0.125, 0.375, 0.625, 0.875])
SHIFT_RATE = 1.6
GAMMA, ALPHA = 0.95, 0.1
K = 10  # independent training seeds


def new_state(K_, E_, rng):
    r = rng.integers(0, N, (K_, E_))
    c = rng.integers(0, N, (K_, E_))
    bad = HAZ[r, c]
    r[bad], c[bad] = 0, 0
    one = np.ones((K_, E_))
    return dict(r=r, c=c, e=one.copy(), h=one.copy(), d=one.copy(),
                alive=np.ones((K_, E_), bool),
                tF=np.zeros((K_, E_), int), tW=np.zeros((K_, E_), int))


def step(st, a, rate):
    alive = st['alive']
    mv = (a < 4) & alive
    st['r'] = np.where(mv, np.clip(st['r'] + DR[a], 0, N - 1), st['r'])
    st['c'] = np.where(mv, np.clip(st['c'] + DC[a], 0, N - 1), st['c'])
    atF = (st['r'] == FOOD[0]) & (st['c'] == FOOD[1])
    atW = (st['r'] == WATER[0]) & (st['c'] == WATER[1])
    cons = (a == 4) & alive
    e0, h0, d0 = st['e'].copy(), st['h'].copy(), st['d'].copy()
    e = e0 - E_DEC * rate
    h = h0 - H_DEC * rate
    eF, wW = cons & atF, cons & atW
    e = np.where(eF, np.minimum(1.0, e + GAIN), e)
    h = np.where(wW, np.minimum(1.0, h + GAIN), h)
    haz = HAZ[st['r'], st['c']]
    d = np.where(haz, d0 - HAZ_DMG,
                 np.where((e > 0.3) & (h > 0.3), np.minimum(1.0, d0 + REGEN), d0))
    e, h, d = [np.where(alive, x, o) for x, o in ((e, e0), (h, h0), (d, d0))]
    dead = alive & ((e <= 0) | (h <= 0) | (d <= 0))
    e, h, d = np.clip(e, 0, 1), np.clip(h, 0, 1), np.clip(d, 0, 1)
    st['e'], st['h'], st['d'] = e, h, d
    st['tF'] = np.where(eF, 0, st['tF'] + 1)
    st['tW'] = np.where(wW, 0, st['tW'] + 1)
    in_v = alive & ~dead & (e >= THETA) & (h >= THETA) & (d >= THETA)
    D0 = (1 - e0) ** 2 + (1 - h0) ** 2 + (1 - d0) ** 2
    D1 = (1 - e) ** 2 + (1 - h) ** 2 + (1 - d) ** 2
    r_aff = np.where(alive, D0 - D1 - 5.0 * dead, 0.0)
    intake = (e - e0).clip(0) * eF + (h - h0).clip(0) * wW
    r_clk = np.where(alive, 2.0 * intake - 0.5 * haz - 5.0 * dead, 0.0)
    st['alive'] = alive & ~dead
    return r_aff, r_clk, dead, in_v, alive


def obs(kind, st, sigma, rng, ablate=False):
    pos = st['r'] * N + st['c']
    if kind == 'aff':
        if ablate:
            eb = np.full(pos.shape, 2); hb = np.full(pos.shape, 2); db = np.full(pos.shape, 3)
        else:
            eb = np.digitize(st['e'] + rng.normal(0, sigma, pos.shape), EDGES)
            hb = np.digitize(st['h'] + rng.normal(0, sigma, pos.shape), EDGES)
            db = np.digitize(st['d'] + rng.normal(0, sigma, pos.shape), EDGES)
        return pos * 64 + eb * 16 + hb * 4 + db
    tf = np.digitize(st['tF'], [10, 20, 35])
    tw = np.digitize(st['tW'], [8, 16, 25])
    return pos * 16 + tf * 4 + tw


NS = {'aff': N * N * 64, 'clk': N * N * 16}


def greedy(Q, s, rng):
    k = np.arange(Q.shape[0])[:, None]
    q = Q[k, s] + rng.random(Q[k, s].shape) * 1e-6
    return q.argmax(-1)


def evaluate(kind, Q, sigma, rate, E, rng, ablate=False, record=False, policy='learned'):
    st = new_state(K, E, rng)
    V = np.zeros((K, E)); surv = np.zeros((K, E))
    rec = dict(e=[], h=[], d=[], ye=[], yh=[], yd=[], alive=[], r=[], c=[], a=[]) if record else None
    for t in range(T):
        if policy == 'random':
            a = rng.integers(0, A, (K, E))
        else:
            a = greedy(Q, obs(kind, st, sigma, rng, ablate), rng)
        if record:
            for key in ('e', 'h', 'd', 'r', 'c'):
                rec[key].append(st[key].copy())
            rec['alive'].append(st['alive'].copy())
            rec['a'].append(a.copy())
            if kind == 'aff':
                rec['ye'].append(st['e'] + rng.normal(0, sigma, (K, E)))
                rec['yh'].append(st['h'] + rng.normal(0, sigma, (K, E)))
                rec['yd'].append(st['d'] + rng.normal(0, sigma, (K, E)))
        _, _, _, in_v, _ = step(st, a, rate)
        V += in_v
    V /= T
    surv = st['alive'].astype(float)
    if record:
        rec = {k_: np.array(v) for k_, v in rec.items() if len(v)}
    return V, surv, rec


def train(kind, sigma, episodes, rng, log=True, eval_every=100):
    Q = np.zeros((K, NS[kind], A))
    kk = np.arange(K)[:, None]
    curve = []
    for ep in range(episodes):
        eps = max(0.05, 1.0 - 0.95 * ep / (0.6 * episodes))
        st = new_state(K, 1, rng)
        s = obs(kind, st, sigma, rng)
        for t in range(T):
            a = np.where(rng.random((K, 1)) < eps, rng.integers(0, A, (K, 1)), greedy(Q, s, rng))
            r_aff, r_clk, dead, _, alive_b = step(st, a, 1.0)
            r = r_aff if kind == 'aff' else r_clk
            s2 = obs(kind, st, sigma, rng)
            target = r + GAMMA * Q[kk, s2].max(-1) * (~dead)
            cur = Q[kk, s, a]
            Q[kk, s, a] = np.where(alive_b, cur + ALPHA * (target - cur), cur)
            s = s2
        if ep % eval_every == 0 or ep == episodes - 1:
            V, _, _ = evaluate(kind, Q, sigma, 1.0, 30, rng)
            curve.append((ep, V.mean(1)))
            if log:
                print(f"  [{kind} sigma={sigma}] ep {ep:5d}  V={V.mean():.3f}", flush=True)
    return Q, curve


# --------------------------------------------------------------- metrics
def ci95(x):
    x = np.asarray(x, float)
    if x.size < 2:
        return float(x.mean()), 0.0
    return float(x.mean()), float(2.262 * x.std(ddof=1) / np.sqrt(x.size))


def s1_accuracy(rec, kind, report_sigma=None, rng=None):
    """1 - RMSE(estimate)/RMSE(time-mean predictor), per seed."""
    out = []
    alive = rec['alive']
    for k in range(K):
        errs, refs = [], []
        for key, ykey in (('e', 'ye'), ('h', 'yh'), ('d', 'yd')):
            x = rec[key][:, k, :][alive[:, k, :]]
            if kind == 'aff':
                y = rec[ykey][:, k, :][alive[:, k, :]]
                est = CENTERS[np.digitize(y, EDGES)]
            else:
                est = np.clip(x + rng.normal(0, report_sigma, x.shape), 0, 1)
            errs.append(((est - x) ** 2).mean())
            refs.append(((x.mean() - x) ** 2).mean())
        out.append(max(0.0, 1 - np.sqrt(np.mean(errs)) / np.sqrt(np.mean(refs))))
    return np.array(out)


def softmax_rows(q, tau):
    z = q / tau
    z = z - z.max(-1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(-1, keepdims=True)


CAND_E = [(1, 2, 3), (1, 3, 3), (0, 2, 3), (0, 3, 3)]  # energy lower than hydration
CAND_H = [(2, 1, 3), (3, 1, 3), (2, 0, 3), (3, 0, 3)]  # hydration lower than energy


def s4_social(Q_obs, demo_rec, rng, n=300, W=6, tau=0.5):
    """Observer infers which variable is lower from demonstrator's W-step action window."""
    accs = []
    for k in range(K):
        kd = (k + 1) % K
        e, h = demo_rec['e'][:, kd, :], demo_rec['h'][:, kd, :]
        r, c, a = demo_rec['r'][:, kd, :], demo_rec['c'][:, kd, :], demo_rec['a'][:, kd, :]
        al = demo_rec['alive'][:, kd, :]
        cands = []
        for t in range(W, T):
            for j in range(e.shape[1]):
                if al[t, j] and al[t - W, j] and abs(e[t - 1, j] - h[t - 1, j]) > 0.2:
                    cands.append((t, j))
        if len(cands) > n:
            cands = [cands[i] for i in rng.choice(len(cands), n, replace=False)]
        correct = 0
        for (t, j) in cands:
            truth = 0 if e[t - 1, j] < h[t - 1, j] else 1
            ll = []
            for cls in (CAND_E, CAND_H):
                tot = 0.0
                for (eb, hb, db) in cls:
                    lp = 0.0
                    for u in range(t - W, t):
                        s = (r[u, j] * N + c[u, j]) * 64 + eb * 16 + hb * 4 + db
                        p = softmax_rows(Q_obs[k, s], tau)
                        lp += np.log(p[a[u, j]] + 1e-9)
                    tot += np.exp(lp)
                ll.append(tot / len(cls))
            pred = int(np.argmax(ll)) if abs(ll[0] - ll[1]) > 1e-12 else int(rng.integers(0, 2))
            correct += (pred == truth)
        accs.append(correct / max(1, len(cands)))
    return np.array(accs)


def meq(s, C, w=(0.25, 0.25, 0.25, 0.25)):
    return 100.0 * C * float(np.dot(w, s))


# ------------------------------------------------------------------ main
def main(episodes=3000, quick=False):
    t0 = time.time()
    rng = np.random.default_rng(2026)
    E = 100 if not quick else 20
    res = {'config': dict(episodes=episodes, seeds=K, horizon=T, eval_episodes=E,
                          shift_rate=SHIFT_RATE, theta=THETA)}
    traces = {}

    # reference: random policy
    V0n, S0n, _ = evaluate('aff', None, 0, 1.0, E, rng, policy='random')
    V0s, S0s, _ = evaluate('aff', None, 0, SHIFT_RATE, E, rng, policy='random')
    res['random'] = dict(V_nom=ci95(V0n.mean(1)), V_shift=ci95(V0s.mean(1)),
                         surv_nom=ci95(S0n.mean(1)), surv_shift=ci95(S0s.mean(1)))
    V0 = float(V0n.mean())
    print('random V0 =', round(V0, 3), flush=True)

    # clocked non-affective baseline
    print('training clocked baseline', flush=True)
    Qc, curve_c = train('clk', 0.0, episodes, rng)
    Vcn, Scn, rec_cn = evaluate('clk', Qc, 0, 1.0, E, rng, record=True)
    Vcs, Scs, _ = evaluate('clk', Qc, 0, SHIFT_RATE, E, rng)
    res['clocked'] = dict(V_nom=ci95(Vcn.mean(1)), V_shift=ci95(Vcs.mean(1)),
                          surv_nom=ci95(Scn.mean(1)), surv_shift=ci95(Scs.mean(1)))
    Vb_s = Vcs.mean(1)

    # faker: clocked policy + unused report channel
    s1_fake = s1_accuracy(rec_cn, 'report', report_sigma=0.10, rng=rng)

    # affective agents at several interoceptive noise levels
    sigmas = [0.05, 0.15, 0.30, 0.50]
    res['affective'] = {}
    curves = {}
    demo_rec = None
    Qs = {}
    for sg in sigmas:
        print(f'training affective sigma={sg}', flush=True)
        Qa, curve_a = train('aff', sg, episodes, rng)
        Qs[sg] = Qa
        curves[sg] = curve_a
        Van, San, rec_an = evaluate('aff', Qa, sg, 1.0, E, rng, record=True)
        Vas, Sas, _ = evaluate('aff', Qa, sg, SHIFT_RATE, E, rng)
        Vab, Sab, _ = evaluate('aff', Qa, sg, 1.0, E, rng, ablate=True)
        Vabs, _, _ = evaluate('aff', Qa, sg, SHIFT_RATE, E, rng, ablate=True)
        if sg == 0.05:
            demo_rec = rec_an
            traces['aff_example'] = {k_: v[:, 0, 0].tolist() for k_, v in rec_an.items()
                                     if k_ in ('e', 'h', 'd', 'r', 'c', 'a', 'alive')}
        s1 = s1_accuracy(rec_an, 'aff')
        Va = Van.mean(1)
        s2 = np.clip((Va - V0) / (1 - V0), 0, 1)
        s3 = np.clip((Vas.mean(1) - Vb_s) / (1 - Vb_s), 0, 1)
        s4 = np.clip((s4_social(Qa, demo_rec if demo_rec is not None else rec_an, rng) - 0.5) / 0.5, 0, 1)
        Pab = Vab.mean(1)
        C = np.clip((Va - Pab) / np.maximum(Va, 1e-9), 0, 1)
        meqs = np.array([meq((s1[i], s2[i], s3[i], s4[i]), C[i]) for i in range(K)])
        res['affective'][str(sg)] = dict(
            V_nom=ci95(Va), V_shift=ci95(Vas.mean(1)), V_ablated=ci95(Pab),
            V_ablated_shift=ci95(Vabs.mean(1)),
            surv_nom=ci95(San.mean(1)), surv_shift=ci95(Sas.mean(1)), surv_ablated=ci95(Sab.mean(1)),
            s1=ci95(s1), s2=ci95(s2), s3=ci95(s3), s4=ci95(s4), C=ci95(C), MEQ=ci95(meqs))

    # clocked and faker MEQ profiles (gate closed: no causal interoceptive loop)
    Vc = Vcn.mean(1)
    s2c = np.clip((Vc - V0) / (1 - V0), 0, 1)
    res['clocked'].update(s1=(0.0, 0.0), s2=ci95(s2c), s3=(0.0, 0.0), s4=(0.0, 0.0), C=(0.0, 0.0), MEQ=(0.0, 0.0))
    res['faker'] = dict(V_nom=res['clocked']['V_nom'], V_shift=res['clocked']['V_shift'],
                        s1=ci95(s1_fake), s2=ci95(s2c), s3=(0.0, 0.0), s4=(0.0, 0.0), C=(0.0, 0.0), MEQ=(0.0, 0.0))
    res['random'].update(s1=(0.0, 0.0), s2=(0.0, 0.0), s3=(0.0, 0.0), s4=(0.0, 0.0), C=(0.0, 0.0), MEQ=(0.0, 0.0))

    res['curves'] = {'clocked': [(ep, float(v.mean())) for ep, v in curve_c]}
    for sg in sigmas:
        res['curves'][f'aff_{sg}'] = [(ep, float(v.mean())) for ep, v in curves[sg]]
    res['curves_seed'] = {'clocked': [v.tolist() for ep, v in curve_c],
                          'aff_0.05': [v.tolist() for ep, v in curves[0.05]],
                          'episodes': [ep for ep, v in curve_c]}
    res['traces'] = traces
    res['runtime_s'] = round(time.time() - t0, 1)
    with open('results.json', 'w') as f:
        json.dump(res, f, indent=1, default=float)
    print('done in', res['runtime_s'], 's')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        main(episodes=300, quick=True)
    else:
        main(episodes=int(sys.argv[1]) if len(sys.argv) > 1 else 3000)
