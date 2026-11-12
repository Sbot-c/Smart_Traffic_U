"""Light-theme figures for the requirement specification v2.1."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_H = os.path.dirname(os.path.abspath(__file__))
DATA, OUT = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else \
    (os.path.join(os.path.dirname(os.path.dirname(_H)), "data"), _H)
COL = {"Fixed-time": "#d99a00", "Actuated": "#1baf7a", "Q-learning": "#2a78d6", "SARSA": "#eb6834"}
ORDER = ["normal", "rush_hour", "incident", "rain", "sandstorm", "event_surge", "bus_corridor"]
LAB = ["Normal", "Rush hour", "Incident", "Rain", "Sandstorm", "Event surge", "Bus corridor"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e4dc",
                     "axes.axisbelow": True})

kpi = pd.read_csv(f"{DATA}/kpi_results.csv")
g = kpi.groupby(["scenario", "policy"]).avg_delay_s.agg(["mean", "std"])
fig, ax = plt.subplots(figsize=(7.2, 2.5))
w = 0.2
for k, pol in enumerate(COL):
    m = [g.loc[(s, pol), "mean"] for s in ORDER]
    sd = [g.loc[(s, pol), "std"] for s in ORDER]
    ax.bar(np.arange(7) + (k - 1.5) * w, m, w * 0.92, yerr=sd, color=COL[pol], label=pol,
           capsize=1.5, error_kw=dict(lw=0.6, ecolor="#888"))
ax.set_xticks(np.arange(7), LAB)
ax.set_ylabel("average delay (s/veh)")
ax.legend(frameon=False, ncol=4, loc="upper left")
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_delay.png", dpi=220)

s = pd.read_csv(f"{DATA}/ext_sensing_summary.csv")
fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.4), sharey=False)
for ax, sc, nm in zip(axes, ["rush_hour", "event_surge"], ["Rush hour", "Event surge"]):
    d = s[(s.scenario == sc) & (~s.noise)]
    ft = d[d.policy == "Fixed-time"].avg_delay_s.iloc[0]
    ax.axhline(ft, color=COL["Fixed-time"], ls="--", lw=1.2, label="Fixed-time (no detectors)")
    for pol in ["Actuated", "Q-learning", "SARSA"]:
        e = d[d.policy == pol].sort_values("bias")
        ax.plot(e.bias, e.avg_delay_s, "-o", color=COL[pol], lw=1.8, ms=3.5, label=pol)
    ax.axvline(1.0, color="#999", lw=0.8, ls=":")
    ax.set_title(nm, fontsize=9, loc="left")
    ax.set_xlabel("queue-count bias (1 = exact; 2 = two phones per person)", fontsize=7.5)
axes[0].set_ylabel("average delay (s/veh)")
axes[1].legend(frameon=False, fontsize=7.5, loc="upper right")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_sensing.png", dpi=220)
print("figures written")
