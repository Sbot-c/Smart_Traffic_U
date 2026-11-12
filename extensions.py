"""
extensions.py - studies that close the five gaps listed in the project review.

  1. AV speed control and lane selection   -> V2I speed advisory (GLOSA) + AV lane selection
  2. Bus passenger load balancing          -> load KPIs + an even-headway holding rule
  3. Proxy safety model                    -> sensitivity of the policy ranking to the
                                              conflict-model parameters
  4. Fairness of the comparison            -> actuated controller tuned on validation runs;
                                              RL reported as a mean over training runs
  5. RL safety at high demand              -> pedestrian max-wait guard (safety supervisor)

Every study uses the deployed Q-tables (no retraining), the held-out TEST seeds for
reporting and, where something is tuned, the VALIDATION seeds for choosing. The new
mechanisms are switched on through scenario overrides, so the published results are
unchanged when they are off.

Run:  python extensions.py        (about 3 minutes) -> data/ext_*.csv
"""
from __future__ import annotations

import itertools
import os
import time

import numpy as np
import pandas as pd

import agents as A
import config as C
import experiments as X
import traffic_env as TE

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
N_TEST = 8
N_VAL = 3
STRESS = dict(demand_mult=1.6, ped_mult=1.6, pax_mult=round(1.6 ** 2.24, 1), profile="peak",
              corridor_mult=1.15)      # beyond rush hour: the stress-test condition
KEEP = ["avg_delay_s", "throughput_vph", "avg_stops", "co2_kg", "nox_g", "conflicts_per_1000",
        "safety_score", "jaywalks", "ped_avg_wait_s", "bus_on_time_pct", "bus_avg_delay_s",
        "av_avg_delay_s", "human_avg_delay_s", "bus_load_mean", "bus_load_cv",
        "bus_load_p95_pct", "pax_left_behind", "ext_glides", "ext_glide_s", "ext_bypass",
        "ext_ped_serves"]


def system(models, pol, sc):
    return A.make_system(pol, X.policy_tables(models, pol, sc), plan=A.webster_plan("normal"))


def run(sysm, sc, seeds, ov=None):
    rows = []
    for j in seeds:
        k = sysm.run_episode(sc, seed=j, overrides=ov).kpis()
        rows.append({c: k[c] for c in KEEP} | dict(seed=j))
    return rows


TEST = [X.EVAL_SEED0 + j for j in range(N_TEST)]
VAL = [X.VAL_SEED0 + j for j in range(N_VAL)]


# --------------------------------------------------------------------------------------
# 1. AV speed advisory and lane selection
# --------------------------------------------------------------------------------------
AV_VARIANTS = {"Baseline": {}, "+ speed advisory": {"av_glosa": True},
               "+ lane selection": {"av_lane": True},
               "+ both": {"av_glosa": True, "av_lane": True}}


def study_av(models):
    rows = []
    for sc in C.SCENARIO_ORDER:
        sysm = system(models, "Q-learning", sc)
        for name, ov in AV_VARIANTS.items():
            for r in run(sysm, sc, TEST, ov):
                rows.append(r | dict(scenario=sc, variant=name))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 2. Bus load balancing
# --------------------------------------------------------------------------------------
class HeadwayBus:
    """Even-headway holding: hold at a stop when the bus is running close behind the
    previous one (gap < 60 % of the timetable headway), so the gap - and the passenger load
    it carries - evens out. A standard transit-operations rule, used as a benchmark for the
    learned bus agents."""

    def __init__(self, ratio=0.6):
        self.ratio = ratio
        self.log = None

    def begin(self, train, eps, record=False):
        self.log = [] if record else None

    def decide(self, env, v, i, a):
        b = v.bus
        last = env.last_dep_stop.get((b.direction, i))
        hw = env.bus_headway.get(b.direction, 360.0)
        if last is not None and (b.t_ready - last) < self.ratio * hw:
            return 2
        return 0

    def learn_arrival(self, env, v, dev, terminal=False):
        pass


BUS_VARIANTS = ["No priority, no holding", "Always request priority",
                "Even-headway holding rule", "RL bus agents (deployed)"]


def study_bus(models):
    rows = []
    for sc in ["normal", "rush_hour", "bus_corridor"]:
        for name in BUS_VARIANTS:
            sysm = system(models, "Q-learning", sc)
            if name == BUS_VARIANTS[0]:
                sysm.bus = A.FixedBus("none")
            elif name == BUS_VARIANTS[1]:
                sysm.bus = A.FixedBus("always")
            elif name == BUS_VARIANTS[2]:
                sysm.bus = HeadwayBus()
            for r in run(sysm, sc, TEST):
                rows.append(r | dict(scenario=sc, variant=name))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 3. Sensitivity of the safety ranking to the conflict model
# --------------------------------------------------------------------------------------
SAFETY_VARIANTS = {"Base model": {},
                   "Red-light running × 0.5": {"rlr_mult": 0.5},
                   "Red-light running × 2": {"rlr_mult": 2.0},
                   "Rear-end (stops) × 0.5": {"stop_conf_mult": 0.5},
                   "Rear-end (stops) × 2": {"stop_conf_mult": 2.0},
                   "Jaywalking × 0.5": {"jay_conf_mult": 0.5},
                   "Jaywalking × 2": {"jay_conf_mult": 2.0}}


def study_safety_sensitivity(models, n=4):
    rows = []
    for sc in C.SCENARIO_ORDER:
        for pol in A.POLICY_ORDER:
            sysm = system(models, pol, sc)
            for name, ov in SAFETY_VARIANTS.items():
                for r in run(sysm, sc, TEST[:n], ov):
                    rows.append(dict(scenario=sc, policy=pol, variant=name, seed=r["seed"],
                                     conflicts_per_1000=r["conflicts_per_1000"]))
    return pd.DataFrame(rows)


def ranking_stability(df):
    """Per scenario: is the safest policy and the full ranking the same as the base model?"""
    g = df.groupby(["scenario", "variant", "policy"]).conflicts_per_1000.mean().reset_index()
    out = []
    for sc, sub in g.groupby("scenario"):
        base = sub[sub.variant == "Base model"].sort_values("conflicts_per_1000").policy.tolist()
        for v, s2 in sub.groupby("variant"):
            r = s2.sort_values("conflicts_per_1000").policy.tolist()
            out.append(dict(scenario=sc, variant=v, safest=r[0], same_safest=r[0] == base[0],
                            same_ranking=r == base, ranking=" < ".join(r)))
    return pd.DataFrame(out)


# --------------------------------------------------------------------------------------
# 4. Tuned actuated controller
# --------------------------------------------------------------------------------------
class TunedActuated(A.ActuatedSignals):
    def __init__(self, max_green=60.0, ratio=2.0, offset=4.0, ped_after=25.0):
        self.p = dict(max_green=max_green, ratio=ratio, offset=offset, ped_after=ped_after)

    def act(self, env):
        p = self.p
        out = []
        for i in range(C.N_INT):
            ph, el, qg, qr, arr_g, ped, spill, q = env.signal_obs(i)
            qg += env.sensed(sum(env.approaching(i, a, 4.0) for a in range(4)
                      if C.PHASE_OF_APPROACH[a] == ph))
            if el < C.MIN_GREEN:
                out.append(0)
            elif el >= p["max_green"]:
                out.append(1)
            elif qg == 0 and qr > 0:
                out.append(1)
            elif qr >= p["ratio"] * qg + p["offset"]:
                out.append(1)
            elif ped and el >= p["ped_after"]:
                out.append(1)
            else:
                out.append(0)
        return out


GRID = dict(max_green=[45.0, 60.0, 75.0], ratio=[1.5, 2.0, 3.0], offset=[2.0, 4.0, 6.0],
            ped_after=[20.0, 25.0, 35.0])


def study_tuned_actuated(models):
    """Tune the four actuated parameters per scenario on VALIDATION seeds (objective = the
    same weighted score as the RL reward uses: delay first), report on TEST seeds with both
    static and cooperative RL routing."""
    combos = list(itertools.product(*GRID.values()))
    keys = list(GRID)
    tune_rows, rows = [], []
    for sc in C.SCENARIO_ORDER:
        best, best_v = None, np.inf
        for combo in combos:
            p = dict(zip(keys, combo))
            sysm = A.make_system("Actuated", plan=A.webster_plan("normal"))
            sysm.signals = TunedActuated(**p)
            v = np.mean([sysm.run_episode(sc, seed=s).kpis()["avg_delay_s"] for s in VAL])
            if v < best_v:
                best, best_v = p, v
        tune_rows.append(dict(scenario=sc, **best, validation_delay_s=best_v))
        for routing in ["Static", "Cooperative RL"]:
            for label, params in [("Actuated (untuned)", {}), ("Actuated (tuned)", best)]:
                ro = None if routing == "Static" else "qlearning"
                sysm = A.make_system("Actuated", X.policy_tables(models, "Q-learning", sc),
                                     plan=A.webster_plan("normal"), routing_override=ro)
                sysm.signals = TunedActuated(**params)
                for r in run(sysm, sc, TEST):
                    rows.append(r | dict(scenario=sc, controller=label, routing=routing))
            if routing == "Cooperative RL":
                sysm = system(models, "Q-learning", sc)
                for r in run(sysm, sc, TEST):
                    rows.append(r | dict(scenario=sc, controller="Q-learning (deployed)",
                                         routing=routing))
    return pd.DataFrame(tune_rows), pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 5. Pedestrian max-wait guard (safety supervisor on top of the RL signals)
# --------------------------------------------------------------------------------------
GUARDS = {"Q-learning": 0.0, "Q-learning + guard 30 s": 30.0,
          "Q-learning + guard 26 s": 26.0, "Q-learning + guard 22 s": 22.0}


def study_guard(models):
    rows = []
    conds = [(sc, None, C.SCENARIOS[sc]["label"]) for sc in C.SCENARIO_ORDER]
    conds.append(("normal", STRESS, "Stress: peak at 1.6 × demand"))
    for sc, ov0, label in conds:
        tables_sc = sc if ov0 is None else "normal"
        for name, g in GUARDS.items():
            sysm = system(models, "Q-learning", tables_sc) if ov0 is None else \
                A.make_system("Q-learning", X.policy_tables(models, "Q-learning", "normal",
                                                            adapted=False),
                              plan=A.webster_plan("normal"))
            ov = dict(ov0 or {})
            if g:
                ov["ped_guard"] = g
            for r in run(sysm, sc, TEST, ov or None):
                rows.append(r | dict(condition=label, variant=name))
        sysm = system(models, "Actuated", tables_sc)
        for r in run(sysm, sc, TEST, ov0):
            rows.append(r | dict(condition=label, variant="Actuated package (reference)"))
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    models = X.load_models(os.path.join(DATA, "q_tables.npz"))
    X.apply_selection(models, os.path.join(DATA, "model_selection.csv"))
    study_av(models).to_csv(os.path.join(DATA, "ext_av_speed_lane.csv"), index=False)
    print(f"AV study done {time.time() - t0:.0f}s")
    study_bus(models).to_csv(os.path.join(DATA, "ext_bus_load.csv"), index=False)
    print(f"bus study done {time.time() - t0:.0f}s")
    s = study_safety_sensitivity(models)
    s.to_csv(os.path.join(DATA, "ext_safety_sensitivity.csv"), index=False)
    ranking_stability(s).to_csv(os.path.join(DATA, "ext_safety_ranking.csv"), index=False)
    print(f"safety sensitivity done {time.time() - t0:.0f}s")
    tune, ta = study_tuned_actuated(models)
    tune.to_csv(os.path.join(DATA, "ext_actuated_tuning.csv"), index=False)
    ta.to_csv(os.path.join(DATA, "ext_tuned_actuated.csv"), index=False)
    print(f"tuned actuated done {time.time() - t0:.0f}s")
    study_guard(models).to_csv(os.path.join(DATA, "ext_ped_guard.csv"), index=False)
    print(f"guard study done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
