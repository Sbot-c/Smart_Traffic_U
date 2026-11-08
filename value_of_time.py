"""
value_of_time.py - turn saved vehicle-hours (and pedestrian-hours) into money.

Why: delay in seconds is hard to weigh against anything else. Converting it to currency
puts efficiency on the same scale as a cost (hardware, communications) and lets the group
state the trade-off between vehicle delay and pedestrian waiting as a net benefit.

Method (standard in transport cost-benefit analysis):
    value of time saved = hours saved x value of one hour
  * In-vehicle (local personal) travel time is valued at 50 % of the hourly median household
    income; walking / waiting / access time is valued at 100 % of it, i.e. twice as much.
    Source: U.S. DOT, "The Value of Travel Time Savings: Departmental Guidance for Conducting
    Economic Evaluations", Revision 2 (2016 update); the original 1997 guidance is described in
    the Federal Register notice of 12 Nov 1997 (document 97-29718).  The 2015 values reproduced by the FAA benefit-cost guidance
    (Table 1-1): local travel, all purposes, USD 14.10 per hour (plausible range 10.00 - 17.00),
    in 2015 US dollars.
  * AED is pegged to the US dollar at 3.6725 AED per USD.

LIMITS (say these out loud): the dollar figure is a US national value, not a Dubai one. A
Dubai income-based value would be higher, so the default is on the conservative side for
travel time but uncorrected for inflation since 2015. Everything below is a parameter; the
dashboard lets the user override it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C

USD_AED = 3.6725                    # AED per USD (fixed peg)
VOT_USD_2015 = 14.10                # USD per person-hour, local travel, all purposes (USDOT 2016)
VOT_USD_RANGE = (10.00, 17.00)      # plausible range in the same table
VOT_AED_DEFAULT = round(VOT_USD_2015 * USD_AED, 1)          # 51.8 AED per hour
WAIT_MULT_DEFAULT = 2.0             # out-of-vehicle (walking, waiting) time = 100 % vs 50 % of wage
OCCUPANCY_DEFAULT = 1.0             # persons per vehicle; 1.0 = driver only (conservative)
WINDOWS_PER_YEAR_DEFAULT = 500      # two measured peak windows x 250 working days (ASSUMPTION)
WINDOW_H = (C.EPISODE_S - C.WARMUP_S) / 3600.0      # measured part of one episode, hours

SOURCE_TEXT = (
    "Value of time: U.S. DOT, Value of Travel Time Savings, Departmental Guidance, Revision 2 "
    "(2016 update): local travel, all purposes, USD 14.10 per hour in 2015 dollars (range "
    "10.00 - 17.00), i.e. 50 % of hourly median household income; walking and waiting time is "
    "valued at 100 % of income (twice as much). Converted at the AED peg of 3.6725 per USD. "
    "This is a US value, not a Dubai one: replace it if a local value is available.")


def vehicle_hours(df: pd.DataFrame) -> pd.Series:
    """Total vehicle delay in the measured window, in vehicle-hours, per row of a KPI frame."""
    return df["vehicles"] * df["avg_delay_s"] / 3600.0


def savings_table(kpi: pd.DataFrame, baseline: str = "Fixed-time",
                  vot_aed: float = VOT_AED_DEFAULT, occupancy: float = OCCUPANCY_DEFAULT,
                  windows_per_year: float = WINDOWS_PER_YEAR_DEFAULT) -> pd.DataFrame:
    """Vehicle-hours saved against `baseline` (paired by scenario and evaluation seed, which
    share identical arrivals) and their value. One row per scenario x policy."""
    d = kpi.copy()
    d["veh_h"] = vehicle_hours(d)
    key = ["scenario", "seed"]
    base = d[d.policy == baseline].set_index(key)["veh_h"].rename("base_veh_h")
    d = d.join(base, on=key)
    d["saved_veh_h"] = d["base_veh_h"] - d["veh_h"]
    d["saved_person_h"] = d["saved_veh_h"] * occupancy
    d["saved_aed"] = d["saved_person_h"] * vot_aed
    g = d.groupby(["scenario", "policy"]).agg(
        vehicles=("vehicles", "mean"), delay_veh_h=("veh_h", "mean"),
        saved_veh_h=("saved_veh_h", "mean"), saved_veh_h_sd=("saved_veh_h", "std"),
        saved_aed=("saved_aed", "mean"), saved_aed_sd=("saved_aed", "std")).reset_index()
    g["saved_aed_per_year"] = g["saved_aed"] * windows_per_year
    g["saved_aed_per_vehicle"] = g["saved_aed"] / g["vehicles"]
    g["saved_pct"] = 100.0 * g["saved_veh_h"] / (g["delay_veh_h"] + g["saved_veh_h"]).replace(0, np.nan)
    g["_o"] = g.scenario.map({s: i for i, s in enumerate(C.SCENARIO_ORDER)})
    return g.sort_values(["_o", "policy"]).drop(columns="_o").reset_index(drop=True)


# ======================================================================================
# Pedestrian-weight trade-off (data/ped_weight_tradeoff.csv from ped_tradeoff.py)
# ======================================================================================
def tradeoff_summary(sweep: pd.DataFrame, reference: str = "Actuated + Cooperative RL",
                     vot_aed: float = VOT_AED_DEFAULT, occupancy: float = OCCUPANCY_DEFAULT,
                     wait_mult: float = WAIT_MULT_DEFAULT) -> pd.DataFrame:
    """One row per learner x pedestrian weight: mean and spread over training seeds, plus the
    monetary net benefit against `reference` (same cooperative routing, so only the signal
    controller differs).

    net benefit = vehicle-hours saved x occupancy x VOT  -  extra pedestrian-hours x VOT x wait_mult
    (pedestrian-hours that are saved count as a benefit). Positive = better than the reference.
    """
    ref = sweep[sweep.policy == reference]
    rl = sweep[sweep.ped_weight.notna()].copy()
    ref_veh_h = float(vehicle_hours(ref).mean())
    ref_ped_h = float(ref["ped_wait_hours"].mean())
    rl["veh_h"] = vehicle_hours(rl)
    rl["net_aed"] = ((ref_veh_h - rl["veh_h"]) * occupancy
                     - (rl["ped_wait_hours"] - ref_ped_h) * wait_mult) * vot_aed
    # first average over test seeds inside each training run, then over training runs
    run = rl.groupby(["policy", "ped_weight", "train_seed"]).agg(
        delay=("avg_delay_s", "mean"), ped_wait=("ped_avg_wait_s", "mean"),
        ped_over30=("ped_over30_pct", "mean"), jaywalks=("jaywalks", "mean"),
        conflicts=("conflicts_per_1000", "mean"), veh_h=("veh_h", "mean"),
        ped_h=("ped_wait_hours", "mean"), net_aed=("net_aed", "mean")).reset_index()
    out = run.groupby(["policy", "ped_weight"]).agg(
        runs=("train_seed", "nunique"),
        delay=("delay", "mean"), delay_sd=("delay", "std"),
        ped_wait=("ped_wait", "mean"), ped_wait_sd=("ped_wait", "std"),
        ped_over30=("ped_over30", "mean"), jaywalks=("jaywalks", "mean"),
        conflicts=("conflicts", "mean"), veh_h=("veh_h", "mean"), ped_h=("ped_h", "mean"),
        net_aed=("net_aed", "mean"), net_aed_sd=("net_aed", "std")).reset_index()
    out.attrs["reference"] = dict(
        name=reference, delay=float(ref["avg_delay_s"].mean()),
        ped_wait=float(ref["ped_avg_wait_s"].mean()), veh_h=ref_veh_h, ped_h=ref_ped_h)
    out.attrs["refs"] = {
        name: dict(delay=float(g["avg_delay_s"].mean()), ped_wait=float(g["ped_avg_wait_s"].mean()))
        for name, g in sweep[sweep.ped_weight.isna()].groupby("policy")}
    return out


def variant_savings(abl: pd.DataFrame, scenario: str, baseline: str = "Fixed-time + Static",
                    vot_aed: float = VOT_AED_DEFAULT, occupancy: float = OCCUPANCY_DEFAULT,
                    windows_per_year: float = WINDOWS_PER_YEAR_DEFAULT) -> pd.DataFrame:
    """Value of every signal x routing pair against one baseline (ablation_results.csv, study
    'Signal x routing'). Shows where the money comes from: routing, signals, or both."""
    d = abl[(abl.study == "Signal x routing") & (abl.scenario == scenario)].copy()
    d["veh_h"] = vehicle_hours(d)
    base = d[d.variant == baseline].set_index("seed")["veh_h"]
    d["saved_veh_h"] = d["seed"].map(base) - d["veh_h"]
    d["saved_aed"] = d["saved_veh_h"] * occupancy * vot_aed
    g = d.groupby("variant").agg(delay_s=("avg_delay_s", "mean"), saved_veh_h=("saved_veh_h", "mean"),
                                 saved_aed=("saved_aed", "mean"),
                                 saved_aed_sd=("saved_aed", "std")).reset_index()
    g["saved_aed_per_year"] = g["saved_aed"] * windows_per_year
    return g.sort_values("saved_aed", ascending=False).reset_index(drop=True)
