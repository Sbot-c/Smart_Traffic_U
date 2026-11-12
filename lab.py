"""
lab.py - the Simulation Lab behind the dashboard's live what-if view.

Turns a set of operating conditions (demand, pedestrians, weather, AV share, incident,
emergency calls, bus service, event) into simulator overrides, runs the four integrated
policies on identical arrivals, records replays, sweeps one condition to stress-test the
controllers, and maps the conditions to decision-network evidence.

Correlated conditions
---------------------
In the real network the conditions are not independent. When "linked" mode is on, the
dashboard moves the dependent sliders with their drivers. The coefficients come from the
scenario definitions in config.py, so the lab reproduces the evaluated scenarios exactly
when the drivers are set to their values (e.g. rush-hour ramp -> demand 1.30, pedestrians
1.30, bus passengers ~1.8, eastbound corridor +15 %).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import agents as A
import config as C
import experiments as X

# --------------------------------------------------------------------------------------
# Defaults and correlation model
# --------------------------------------------------------------------------------------
DEFAULTS = dict(demand=1.0, ped=1.0, pax=1.0, weather="Clear", peak=False,
                av_share=C.AV_SHARE, incident=False, inc_loss=75, evs=3,
                bus_lane=False, headway_min=6, event=False, glosa=False, av_lane=False,
                ped_guard=0, sense_bias=1.0, sense_noise=False)

PAX_ELASTICITY = math.log(1.8) / math.log(1.3)      # = 2.24: rush hour 1.30 -> 1.80
EV_BY_WEATHER = {"Clear": 3, "Rain": 4, "Sandstorm": 4}   # config: rain/sandstorm scenarios
EV_INCIDENT_EXTRA = 3                                  # config: incident scenario 6 vs 3
EVENT_PAX = 1.5                                        # config: event surge pax_mult
BUS_LANE_HEADWAY_MIN = 3                               # config: bus corridor 180 s
PEAK = dict(demand=1.30, ped=1.30)                     # config: rush hour

# (driver, follower, rule) - shown in the dashboard as the correlation map
LINKS = [
    ("Rush-hour ramp", "Demand", "on → 1.30 × normal, ramping to ~1.45 mid-episode"),
    ("Rush-hour ramp", "Pedestrians", "on → 1.30 × normal"),
    ("Demand", "Pedestrians", "moves 1 : 1 with demand (more trips, more walkers)"),
    ("Demand", "Bus passengers", f"demand ^ {PAX_ELASTICITY:.2f} (crowding grows faster)"),
    ("Event surge", "Bus passengers", "× 1.5 (fans heading home)"),
    ("Weather", "Emergency calls", "rain / sandstorm: 4 per 30 min instead of 3"),
    ("Incident", "Emergency calls", "+3 responders to the crash"),
    ("Bus lane", "Bus headway", "dedicated lane → bus every 3 min"),
]
# applied by the simulator itself whatever the mode (config.WEATHER)
INTERNAL = [
    ("Weather", "Effective demand", "rain × 0.95, sandstorm × 0.85 (trips cancelled)"),
    ("Weather", "Effective pedestrians", "rain × 0.60, sandstorm × 0.30"),
    ("Weather", "Capacity / speed / crash risk", "rain −6 % / −10 % / × 1.71; "
                                                  "sandstorm −12 % / −20 % / × 2.0"),
    ("AV share", "Capacity", "AV platoons need 0.85 of a car's headway"),
]


def snap(x, lo, hi, step):
    x = min(hi, max(lo, x))
    return round(round(x / step) * step, 4)


def linked_ped(demand):
    return snap(demand, 0.25, 4.0, 0.05)


def linked_pax(demand, event=False):
    return snap(demand ** PAX_ELASTICITY * (EVENT_PAX if event else 1.0), 0.5, 3.0, 0.1)


def linked_evs(weather, incident):
    return EV_BY_WEATHER.get(weather, 3) + (EV_INCIDENT_EXTRA if incident else 0)


def effective(cond):
    """Demand / pedestrian levels after the simulator's weather factors."""
    w = C.WEATHER[cond["weather"]]
    peak_mid = C.demand_profile("peak", C.EPISODE_S / 2) if cond["peak"] else 1.0
    return dict(demand=cond["demand"] * w[4], demand_peak=cond["demand"] * w[4] * peak_mid,
                ped=cond["ped"] * w[3], capacity=w[0], speed=w[1], risk=w[2])


# --------------------------------------------------------------------------------------
# Conditions -> simulator overrides
# --------------------------------------------------------------------------------------
def overrides(cond: dict) -> dict:
    ov = dict(weather=cond["weather"], demand_mult=float(cond["demand"]),
              ped_mult=float(cond["ped"]), pax_mult=float(cond["pax"]),
              profile="peak" if cond["peak"] else "flat",
              corridor_mult=1.15 if cond["peak"] else 1.0,
              ev_per_episode=float(cond["evs"]), bus_lane=bool(cond["bus_lane"]),
              bus_headway_s=float(cond["headway_min"]) * 60.0,
              av_share=float(cond["av_share"]), incident=None, event=None)
    if cond["incident"]:
        inc = dict(C.SCENARIOS["incident"]["incident"])
        loss = float(cond["inc_loss"]) / 100.0
        inc["sat_factor"] = max(0.02, 1.0 - loss)
        inc["speed_factor"] = max(0.2, 1.0 - 0.8 * loss)
        ov["incident"] = inc
    if cond["event"]:
        ov["event"] = dict(C.SCENARIOS["event_surge"]["event"])
    ov["av_glosa"] = bool(cond.get("glosa"))            # extensions.py, gap 1
    ov["av_lane"] = bool(cond.get("av_lane"))
    if cond.get("ped_guard"):                            # extensions.py, gap 5
        ov["ped_guard"] = float(cond["ped_guard"])
    if float(cond.get("sense_bias", 1.0)) != 1.0:        # sensing.py: detector / phone bias
        ov["sense_bias"] = float(cond["sense_bias"])
    if cond.get("sense_noise"):
        ov["sense_noise"] = True
    return ov


def _system(models, pol):
    return A.make_system(pol, X.policy_tables(models, pol, "normal", adapted=False),
                         plan=A.webster_plan("normal"))


def run(models, cond: dict, n_seeds=3, policies=None) -> pd.DataFrame:
    """KPIs of every policy under `cond` (identical arrivals per seed)."""
    ov = overrides(cond)
    rows = []
    for pol in policies or A.POLICY_ORDER:
        sysm = _system(models, pol)
        for j in range(n_seeds):
            k = sysm.run_episode("normal", seed=X.EVAL_SEED0 + j, overrides=ov).kpis()
            k.update(policy=pol, seed=j)
            rows.append(k)
    return pd.DataFrame(rows)


def record(models, cond: dict, policies, minutes=20) -> list[dict]:
    """Replay payloads (seed 0) for the canvas, one per policy."""
    ov = overrides(cond)
    out = []
    for pol in policies:
        env = _system(models, pol).run_episode("normal", seed=X.EVAL_SEED0, record=True,
                                               overrides=ov)
        p = X.animation_payload(env, "normal", pol, minutes * 60.0)
        p["incident"] = ov["incident"]
        p["event"] = bool(cond["event"])
        p["bus_lane"] = bool(cond["bus_lane"])
        out.append(p)
    return out


SWEEPS = {
    "demand": ("Demand (× normal)", [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]),
    "ped": ("Pedestrians (× normal)", [0.25, 0.5, 1.0, 2.0, 3.0, 4.0]),
    "av_share": ("Autonomous-vehicle share", [0.0, 0.1, 0.25, 0.4, 0.6, 0.8]),
    "inc_loss": ("Incident capacity loss (%)", [0, 25, 50, 75, 90]),
}


def sweep(models, cond: dict, var: str, linked: bool, n_seeds=2) -> pd.DataFrame:
    """Re-run every policy along one condition, others held at `cond`. In linked mode a
    demand sweep also moves pedestrians and bus passengers, as the sliders would."""
    rows = []
    for v in SWEEPS[var][1]:
        c = dict(cond)
        c[var] = v
        if var == "demand" and linked:
            c["ped"] = linked_ped(v)
            c["pax"] = linked_pax(v, c["event"])
        if var == "inc_loss":
            c["incident"] = v > 0
            if linked:
                c["evs"] = linked_evs(c["weather"], c["incident"])
        df = run(models, c, n_seeds=n_seeds)
        df["x"] = v
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------------------
# Decision-network evidence and multi-criteria score
# --------------------------------------------------------------------------------------
def dn_evidence(cond: dict) -> dict:
    """Map the conditions onto the decision network's chance nodes. Density thresholds sit
    halfway between the levels used to learn the CPTs (Low 0.75, Medium 1.0, High 1.30 with
    the peak ramp); pedestrian thresholds are geometric midpoints of 0.5 / 1 / 4."""
    eff = effective(cond)
    d = eff["demand"] * (1.15 if cond["peak"] else 1.0)
    density = "High" if (d >= 1.15 or cond["incident"] or cond["event"]) else \
        "Low" if d < 0.875 else "Medium"
    pf = eff["ped"] * (4.0 if cond["event"] else 1.0)
    ped = "High" if pf >= 2.0 else "Low" if pf < 0.71 else "Medium"
    return {"Weather": cond["weather"], "Traffic density": density, "Pedestrian flow": ped}


def scores(g: pd.DataFrame, weights: dict) -> pd.Series:
    """Weighted-sum score 0-100 over delay, safety score and CO2 (min-max within the
    four policies), the same rule as the KPI tab."""
    out = pd.Series(0.0, index=g.index)
    for col, key, low in [("avg_delay_s", "U_efficiency", True),
                          ("safety_score", "U_safety", False),
                          ("co2_kg", "U_environment", True)]:
        lo, hi = g[col].min(), g[col].max()
        rng = (hi - lo) or 1.0
        u = (hi - g[col]) / rng if low else (g[col] - lo) / rng
        out += weights[key] * 100 * u
    return out
