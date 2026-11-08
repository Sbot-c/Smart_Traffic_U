"""
ped_tradeoff.py - how much vehicle delay does it cost to protect pedestrians?

The signal reward is  -(delay + W_PED x pedestrian-wait + 300 x conflicts + 0.2 x CO2) / 100.
W_PED (config.py, default 0.5) is a *design choice* the city makes: how many vehicle-seconds is
one pedestrian-second worth?  This script retrains the RL system from scratch in rush hour at a
range of W_PED values and records vehicle delay and pedestrian wait on the held-out test
episodes, so the trade-off can be drawn as a frontier.

Protocol (same as the main pipeline, experiments.py):
  * pre-train on a round-robin of all scenarios, then adapt to rush hour
  * choose general vs adapted tables on VALIDATION episodes (never on test episodes)
  * evaluate on the same TEST seeds and arrivals for every weight
  * repeat with several training seeds - RL results vary from run to run

    python ped_tradeoff.py              # full sweep, Q-learning (about 20 min on one core)
    python ped_tradeoff.py --quick      # smoke test (about 1 min)
    python ped_tradeoff.py --sarsa      # add SARSA
Writes data/ped_weight_tradeoff.csv (one row per weight x algorithm x training seed x test seed).
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

import agents as A
import config as C
import experiments as X

WEIGHTS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)     # 0.5 = the value used everywhere else
TRAIN_SEEDS = (0, 1, 2)
SCENARIO = "rush_hour"
KEEP = ["avg_delay_s", "ped_avg_wait_s", "ped_wait_hours", "ped_served", "ped_over30_pct", "jaywalks", "conflicts_per_1000",
        "co2_kg", "idle_hours", "vehicles", "throughput_vph", "bus_on_time_pct",
        "safety_score"]


def run_one(policy: str, weight: float, train_seed: int, cfg: dict, plan) -> list[dict]:
    C.W_PED = float(weight)                     # read at run time by traffic_env.step
    try:
        model = X.train_algorithm(policy, cfg, seed=train_seed, verbose=False,
                                  adapt_scenarios=[SCENARIO])
        models = {policy: model}
        val = {}
        for ver, ad in (("general", False), ("adapted", True)):
            s = A.make_system(policy, X.policy_tables(models, policy, SCENARIO, adapted=ad),
                              plan=plan)
            val[ver] = np.mean([s.run_episode(SCENARIO, seed=X.VAL_SEED0 + j).kpis()
                                ["avg_delay_s"] for j in range(cfg["eval_seeds"])])
        pick = "adapted" if val["adapted"] <= val["general"] else "general"
        sysm = A.make_system(policy, X.policy_tables(models, policy, SCENARIO,
                                                     adapted=(pick == "adapted")), plan=plan)
        rows = []
        for j in range(cfg["eval_seeds"]):
            k = sysm.run_episode(SCENARIO, seed=X.EVAL_SEED0 + j).kpis()
            rows.append(dict(policy=policy, ped_weight=weight, train_seed=train_seed,
                             test_seed=j, tables=pick, **{c: k[c] for c in KEEP}))
        return rows
    finally:
        C.W_PED = 0.5


def reference_rows(cfg: dict, plan) -> list[dict]:
    """Same-routing reference controllers in rush hour (cooperative AV routing for all), so the
    comparison isolates the signal controller. Uses the models from the main pipeline."""
    models = X.load_models("data/q_tables.npz")
    X.apply_selection(models, "data/model_selection.csv")
    rows = []
    for sig in ("Fixed-time", "Actuated"):
        sysm = X.dn_system(models, sig, "Cooperative RL", SCENARIO, plan)
        for j in range(cfg["eval_seeds"]):
            k = sysm.run_episode(SCENARIO, seed=X.EVAL_SEED0 + j).kpis()
            rows.append(dict(policy=f"{sig} + Cooperative RL", ped_weight=np.nan,
                             train_seed=-1, test_seed=j, tables="", **{c: k[c] for c in KEEP}))
    return rows


def main(argv):
    quick = "--quick" in argv
    cfg = X.QUICK if quick else X.FULL
    policies = ["Q-learning"] + (["SARSA"] if "--sarsa" in argv else [])
    weights = (0.0, 0.5, 4.0) if quick else WEIGHTS
    seeds = (0,) if quick else TRAIN_SEEDS
    out = "data/ped_weight_tradeoff.csv"
    plan = A.webster_plan("normal")
    rows, t0 = reference_rows(cfg, plan), time.time()
    for pol in policies:
        for w in weights:
            for sd in seeds:
                rows += run_one(pol, w, sd, cfg, plan)
                d = pd.DataFrame(rows)
                last = d[(d.policy == pol) & (d.ped_weight == w)]
                print(f"{pol:10s} W_PED {w:5.2f} seed {sd}: delay {last.avg_delay_s.mean():5.1f} s"
                      f"  ped wait {last.ped_avg_wait_s.mean():5.1f} s  ({time.time() - t0:.0f} s)",
                      flush=True)
                d.round(4).to_csv(out if not quick else "/tmp/ped_quick.csv", index=False)
    print("done")


if __name__ == "__main__":
    main(sys.argv[1:])
