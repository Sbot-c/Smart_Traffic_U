"""
make_figures.py - slide-ready PNGs of the pedestrian trade-off and the value of time.

    python make_figures.py            # writes ./figures/*.png

Needs only matplotlib + the CSVs in ./data (no Streamlit / Plotly). The dashboard shows the same
numbers interactively; these static versions are for the presentation.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config as C
import value_of_time as V

COL = {"Fixed-time": "#eda100", "Actuated": "#1baf7a", "Q-learning": "#2a78d6", "SARSA": "#eb6834"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e8e6df"
OUT = "figures"


def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#b9b7ae")
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2)


def ped_tradeoff(path_csv="data/ped_weight_tradeoff.csv", policy="Q-learning"):
    sweep = pd.read_csv(path_csv)
    ts = V.tradeoff_summary(sweep)
    s = ts[ts.policy == policy].sort_values("ped_weight")
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw=dict(width_ratios=[1.3, 1]))
    # --- frontier
    a.plot(s.ped_wait, s.delay, ":", color=COL[policy], lw=1.5, zorder=1)
    a.errorbar(s.ped_wait, s.delay, xerr=s.ped_wait_sd.fillna(0), yerr=s.delay_sd.fillna(0),
               fmt="none", ecolor="#b9b7ae", elinewidth=1, capsize=3, zorder=2)
    a.scatter(s.ped_wait, s.delay, s=[180 if w == 0.5 else 90 for w in s.ped_weight],
              color=COL[policy], edgecolor="white", linewidth=1.5, zorder=3)
    # per-label offsets (points) so the points that nearly coincide (0 / 0.25 / 0.5) and the
    # label that would sit on an error bar (4) stay readable
    off = {0.0: ((-4, -19), "center"), 0.25: ((-12, 11), "center"), 0.5: ((10, 15), "center"),
           4.0: ((-14, 0), "right")}
    for r in s.itertuples():
        (dx, dy), ha = off.get(float(r.ped_weight), ((0, 11), "center"))
        a.annotate(f"{r.ped_weight:g}", (r.ped_wait, r.delay), textcoords="offset points",
                   xytext=(dx, dy), ha=ha, va="center", fontsize=11, color=INK2)
    for key, name, c, m in [("Actuated + Cooperative RL", "Actuated signals", COL["Actuated"], "D"),
                            ("Fixed-time + Cooperative RL", "Fixed-time signals", COL["Fixed-time"], "s")]:
        r = ts.attrs["refs"].get(key)
        if r:
            a.scatter([r["ped_wait"]], [r["delay"]], marker=m, s=110, color=c, edgecolor="white",
                      linewidth=1.5, zorder=3, label=f"{name} (same routing)")
    a.scatter([], [], s=90, color=COL[policy], label=f"{policy} signals, labels = W_PED")
    a.set_xlabel("Average pedestrian wait (s)   ← better", color=INK2)
    a.set_ylabel("Average vehicle delay (s/veh)   ↓ better", color=INK2)
    a.set_title("Rush hour: protecting pedestrians vs moving vehicles", loc="left",
                fontsize=14, color=INK, fontweight="bold")
    a.legend(frameon=False, fontsize=10, loc="best")
    _style(a)
    # --- money
    x = range(len(s))
    b.bar(x, s.net_aed, yerr=s.net_aed_sd.fillna(0), capsize=3, color=[
        "#184f95" if w == 0.5 else COL[policy] for w in s.ped_weight], ecolor="#b9b7ae")
    b.axhline(0, color="#b9b7ae", lw=1)
    b.set_xticks(list(x))
    b.set_xticklabels([f"{w:g}" for w in s.ped_weight])
    b.set_xlabel("Pedestrian weight W_PED in the reward", color=INK2)
    b.set_ylabel("Net benefit vs actuated signals\n(AED per peak window)", color=INK2)
    b.set_title("The same sweep in money", loc="left", fontsize=14, color=INK, fontweight="bold")
    _style(b)
    b.grid(axis="x", visible=False)
    fig.text(0.01, 0.01,
             f"Mean ± 1 sd over {int(s.runs.min())} training runs per weight; 8 held-out test episodes each. "
             f"Value of time AED {V.VOT_AED_DEFAULT:g}/h (USDOT), waiting time counted ×{V.WAIT_MULT_DEFAULT:g}.",
             fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/ped_tradeoff_{policy.lower().replace('-', '')}.png", dpi=200)
    plt.close(fig)
    return ts


def value_of_time(scenario="rush_hour"):
    abl = pd.read_csv("data/ablation_results.csv")
    vs = V.variant_savings(abl, scenario).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.barh(vs.variant, vs.saved_aed, xerr=vs.saved_aed_sd.fillna(0), capsize=3,
            color=[COL[v.split(" + ")[0]] for v in vs.variant], ecolor="#b9b7ae")
    for y, (v, sd) in enumerate(zip(vs.saved_aed, vs.saved_aed_sd.fillna(0))):
        ax.text(max(v, 0) + sd + 40, y, f"{v:,.0f}", va="center", fontsize=10, color=INK2)
    ax.axvline(0, color="#b9b7ae", lw=1)
    ax.set_xlabel("AED saved per 25-minute peak window, vs fixed-time signals + static routes",
                  color=INK2)
    ax.set_title(f"Where the money comes from: {C.SCENARIOS[scenario]['label']}", loc="left",
                 fontsize=14, color=INK, fontweight="bold")
    ax.set_xlim(min(0, (vs.saved_aed - vs.saved_aed_sd.fillna(0)).min() * 1.2),
                (vs.saved_aed + vs.saved_aed_sd.fillna(0)).max() * 1.18)
    _style(ax)
    ax.grid(axis="y", visible=False)
    fig.text(0.01, 0.01, f"AED {V.VOT_AED_DEFAULT:g} per person-hour (USD 14.10, U.S. DOT 2016 update, "
             f"x 3.6725), {V.OCCUPANCY_DEFAULT:g} person per vehicle. Mean ± 1 sd over 8 test seeds.",
             fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/value_of_time_{scenario}.png", dpi=200)
    plt.close(fig)
    return vs


if __name__ == "__main__":
    for sc in ("rush_hour", "event_surge"):
        value_of_time(sc)
    if os.path.exists("data/ped_weight_tradeoff.csv"):
        d = pd.read_csv("data/ped_weight_tradeoff.csv")
        for pol in d[d.ped_weight.notna()].policy.unique():
            ped_tradeoff(policy=pol)
    print("figures written to ./figures")
