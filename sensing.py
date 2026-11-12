"""
sensing.py - sensing uncertainty: what if the controllers do not see the true queues?

Every other study gives the signal controllers exact queue counts. Real counts come from
loop detectors, cameras or phone/probe data, and they are wrong in two ways:

* systematic bias (likelihood bias): phone-based counts that assume one phone per person
  over-count when people carry more than one phone (bias > 1); low detector or probe
  penetration under-counts (bias < 1);
* random counting noise (Poisson).

This study runs the DEPLOYED Q-tables (no retraining) and the actuated rule with biased and
noisy queue observations on the 8 held-out test episodes of every scenario. Fixed-time
signals do not use detectors, so they are the unaffected reference. The physics always uses
the true queues; only what the controllers observe changes.

Run:  python sensing.py   (about 2 minutes on 2 cores) -> data/ext_sensing.csv,
                                                   data/ext_sensing_summary.csv
      python sensing.py --summary   (re-summarise the saved runs only)
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

import agents as A
import config as C
import experiments as X

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
TEST = [X.EVAL_SEED0 + j for j in range(8)]
BIASES = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
SENSING_POLICIES = ["Actuated", "Q-learning", "SARSA"]
KEEP = ["avg_delay_s", "conflicts_per_1000", "ped_avg_wait_s", "co2_kg", "throughput_vph"]
BIAS_LABEL = {0.5: "half of vehicles detected", 0.75: "a quarter missed",
              1.0: "exact counts", 1.25: "over-count 25 %",
              1.5: "1.5 phones per person counted as vehicles",
              2.0: "2 phones per person counted as vehicles"}

_MODELS = None


def _models():
    global _MODELS
    if _MODELS is None:
        _MODELS = X.load_models(os.path.join(DATA, "q_tables.npz"))
        X.apply_selection(_MODELS, os.path.join(DATA, "model_selection.csv"))
    return _MODELS


def _job(args):
    sc, pol = args
    m = _models()
    sysm = A.make_system(pol, X.policy_tables(m, pol, sc), plan=A.webster_plan("normal"))
    rows = []
    conds = [(1.0, False)] if pol == "Fixed-time" else \
        [(b, nz) for b in BIASES for nz in (False, True)]
    for b, nz in conds:
        ov = None if (b == 1.0 and not nz) else dict(sense_bias=b, sense_noise=nz)
        for s in TEST:
            k = sysm.run_episode(sc, seed=s, overrides=ov).kpis()
            rows.append({c: k[c] for c in KEEP} | dict(scenario=sc, policy=pol, bias=b,
                                                        noise=nz, seed=s))
    return rows


def run_study(workers: int = 2) -> pd.DataFrame:
    jobs = [(sc, pol) for sc in C.SCENARIO_ORDER for pol in ["Fixed-time"] + SENSING_POLICIES]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(_job, jobs):
            rows.extend(r)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Mean delay per scenario x policy x sensing condition, change against exact sensing
    and whether the fastest policy in that scenario is still the same one."""
    g = df.groupby(["scenario", "policy", "bias", "noise"], as_index=False)[KEEP].mean()
    exact = g[(g.bias == 1.0) & (~g.noise)].set_index(["scenario", "policy"])["avg_delay_s"]
    g["delay_change_pct"] = [100 * (r.avg_delay_s / exact[(r.scenario, r.policy)] - 1)
                             for r in g.itertuples()]
    fixed = g[g.policy == "Fixed-time"].set_index("scenario")["avg_delay_s"]
    best_exact = (g[(g.bias == 1.0) & (~g.noise)].sort_values("avg_delay_s")
                  .groupby("scenario").policy.first())
    out = []
    for (sc, b, nz), sub in g[g.policy != "Fixed-time"].groupby(["scenario", "bias", "noise"]):
        cand = pd.concat([sub, g[(g.policy == "Fixed-time") & (g.scenario == sc)]])
        best = cand.sort_values("avg_delay_s").policy.iloc[0]
        out.append(dict(scenario=sc, bias=b, noise=nz, best_policy=best,
                        same_best=best == best_exact[sc]))
    best = pd.DataFrame(out)
    # the project's verdict: is the best RL package still ahead of the actuated package?
    pv = g.pivot_table(index=["scenario", "bias", "noise"], columns="policy",
                       values="avg_delay_s")
    pv["rl_ahead_of_actuated"] = pv[["Q-learning", "SARSA"]].min(axis=1) < pv["Actuated"]
    ex = pv.xs((1.0, False), level=("bias", "noise"))["rl_ahead_of_actuated"]
    pv["same_verdict"] = [r for r in (pv.rl_ahead_of_actuated.values ==
                                      ex.reindex(pv.index.get_level_values(0)).values)]
    best = best.merge(pv[["rl_ahead_of_actuated", "same_verdict"]].reset_index(),
                      on=["scenario", "bias", "noise"], how="left")
    g = g.merge(best, on=["scenario", "bias", "noise"], how="left")
    g["fixed_time_delay_s"] = g.scenario.map(fixed)
    return g


def main(summary_only: bool = False):
    t0 = time.time()
    if summary_only:
        df = pd.read_csv(os.path.join(DATA, "ext_sensing.csv"))
    else:
        df = run_study()
        df.to_csv(os.path.join(DATA, "ext_sensing.csv"), index=False)
    s = summarise(df)
    s.to_csv(os.path.join(DATA, "ext_sensing_summary.csv"), index=False)
    print(f"sensing study done in {time.time() - t0:.0f} s, {len(df)} episodes")
    nb = s[s.policy != "Fixed-time"].drop_duplicates(["scenario", "bias", "noise"])
    print(f"fastest policy unchanged in {nb.same_best.mean():.0%} of "
          f"{len(nb)} scenario x sensing conditions; RL-vs-actuated verdict unchanged in "
          f"{nb.same_verdict.mean():.0%}")


if __name__ == "__main__":
    import sys
    main(summary_only="--summary" in sys.argv)
