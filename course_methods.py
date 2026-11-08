"""
course_methods.py - decision-science methods from the course, applied to the simulation
results (no new simulation needed).

  Session 1   decision criteria under uncertainty (maximin, maximax, minimax regret,
              Laplace, Hurwicz) over policies x scenarios
  Session 3   risk-averse (exponential) utility across the evaluation episodes
  Sessions 4-5  the formal MDP of each agent (states, actions, transitions, rewards, discount)
  Sessions 9-10 selfish vs cooperative routing: Nash-style outcome vs system outcome and the
              price of anarchy
  Session 11  AHP weights with a consistency check, and a TOPSIS ranking

All functions take the KPI / ablation data frames written by the pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C

POLICIES = ["Fixed-time", "Actuated", "Q-learning", "SARSA"]


# ======================================================================================
# Session 1 - decisions under uncertainty
# ======================================================================================
def payoff_matrix(kpi: pd.DataFrame, measure: str = "avg_delay_s") -> pd.DataFrame:
    """Rows = policies (acts), columns = scenarios (states of nature).
    Payoff = minus the measure, so that larger is better."""
    m = kpi.groupby(["policy", "scenario"])[measure].mean().unstack()
    m = m.reindex(index=POLICIES, columns=C.SCENARIO_ORDER)
    return -m


def uncertainty_criteria(pay: pd.DataFrame, hurwicz_alpha: float = 0.5) -> pd.DataFrame:
    """What each classical criterion would pick. `pay` is a payoff matrix (larger = better).
    Regret = best payoff in the scenario minus this policy's payoff."""
    regret = pay.max(axis=0) - pay
    score = pd.DataFrame({
        "Maximin (pessimist)": pay.min(axis=1),
        "Maximax (optimist)": pay.max(axis=1),
        "Minimax regret": -regret.max(axis=1),
        "Laplace (equal chances)": pay.mean(axis=1),
        f"Hurwicz (alpha = {hurwicz_alpha:g})": hurwicz_alpha * pay.max(axis=1)
        + (1 - hurwicz_alpha) * pay.min(axis=1),
    })
    return score


def criteria_picks(score: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for crit in score.columns:
        best = score[crit].idxmax()
        rows.append(dict(criterion=crit, picks=best, value=float(score.loc[best, crit])))
    return pd.DataFrame(rows)


# ======================================================================================
# Session 3 - risk-averse utility
# ======================================================================================
def certainty_equivalent(delays: np.ndarray, rho: float) -> float:
    """Exponential (constant absolute risk aversion) utility u(d) = -exp(rho * d) over the
    delay d in seconds. The certainty equivalent is the sure delay with the same utility:
    CE = (1 / rho) * ln mean exp(rho * d). rho = 0 gives the plain mean."""
    d = np.asarray(delays, float)
    if rho == 0:
        return float(d.mean())
    m = d.max()                      # numerically stable log-mean-exp
    return float(m + np.log(np.mean(np.exp(rho * (d - m)))) / rho)


def risk_averse_table(kpi: pd.DataFrame, rho: float = 0.10) -> pd.DataFrame:
    rows = []
    for sc in C.SCENARIO_ORDER:
        for pol in POLICIES:
            d = kpi[(kpi.scenario == sc) & (kpi.policy == pol)].avg_delay_s.values
            rows.append(dict(scenario=sc, policy=pol, mean_delay=d.mean(), sd=d.std(ddof=1),
                             worst=d.max(), certainty_equivalent=certainty_equivalent(d, rho)))
    return pd.DataFrame(rows)


# ======================================================================================
# Sessions 4-5 - the formal MDP of each agent
# ======================================================================================
MDP_TABLE = pd.DataFrame([
    dict(agent="Traffic light (x9)",
         states="phase (2) x green-time bucket x queue on green x queue on red x arrivals on "
                "green x pedestrian / spillback flag = 2,250",
         actions="keep phase, change phase (every 5 s; the safety shield masks illegal changes)",
         transitions="simulator: arrivals, discharge, pedestrians, weather (unknown to the agent, "
                     "learned by sampling - model-free)",
         reward="-(vehicle delay + 0.5 x pedestrian delay + 300 x expected conflicts + 0.2 x g CO2) / 100",
         discount="0.90"),
    dict(agent="Autonomous vehicles (fleet)",
         states="remaining moves in x and y, congestion level of the two candidate links = 64",
         actions="next link along x or along y",
         transitions="simulator: congestion evolves with everyone's routing",
         reward="-(travel time + cooperative congestion externality)",
         discount="0.98"),
    dict(agent="Buses (fleet)",
         states="schedule deviation, load, headway, signal ahead, cross-street queue = 144",
         actions="go, request signal priority, hold 20 s",
         transitions="simulator: dwell times, signal state",
         reward="-(passenger-minutes + schedule deviation + priority externality)",
         discount="0.90"),
])


# ======================================================================================
# Sessions 9-10 - Nash outcome vs system optimum, price of anarchy
# ======================================================================================
def price_of_anarchy(abl: pd.DataFrame) -> pd.DataFrame:
    """Selfish AV routing (each vehicle minimises its own travel time: the Nash-like
    outcome the learners settle in) against cooperative routing (reward includes the delay a
    vehicle imposes on others: a proxy for the system optimum). PoA = delay(selfish) /
    delay(cooperative); >= 1 means selfishness costs the system."""
    s = abl[abl.study == "AV routing"].groupby(["scenario", "variant"]).avg_delay_s.mean()
    rows = []
    for sc in ["normal", "incident", "rush_hour"]:
        if (sc, "Selfish RL routing") not in s.index:
            continue
        selfish, coop = s[(sc, "Selfish RL routing")], s[(sc, "Cooperative RL routing")]
        static = s.get((sc, "Static routes"), np.nan)
        rows.append(dict(scenario=sc, static_routes=static, selfish_routing=selfish,
                         cooperative_routing=coop, price_of_anarchy=selfish / coop))
    return pd.DataFrame(rows)


# ======================================================================================
# Session 11 - AHP weights and TOPSIS
# ======================================================================================
CRITERIA = ["Efficiency", "Safety", "Environment"]
# Saaty 1-9 judgements for how much more important the ROW criterion is than the COLUMN
# criterion. ASSUMPTION: the group's stakeholder judgement; chosen to be consistent with the
# decision network's 0.40 / 0.35 / 0.25 weights (efficiency and safety about equally
# important, both moderately more important than emissions). The judgements are deliberately
# not perfectly consistent (safety vs emissions = 1 although safety vs efficiency = 1 and
# efficiency vs emissions = 2), so the consistency check has something to measure.
AHP_MATRIX = np.array([[1.0, 1.0, 2.0],
                       [1.0, 1.0, 1.0],
                       [1 / 2, 1.0, 1.0]])
RANDOM_INDEX = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12}


def ahp(matrix: np.ndarray = AHP_MATRIX):
    """Principal-eigenvector weights, lambda_max, consistency index and ratio.
    CR < 0.10 is the usual acceptance threshold."""
    vals, vecs = np.linalg.eig(matrix)
    k = int(np.argmax(vals.real))
    w = np.abs(vecs[:, k].real)
    w = w / w.sum()
    n = matrix.shape[0]
    lam = float(vals[k].real)
    ci = (lam - n) / (n - 1)
    cr = ci / RANDOM_INDEX[n] if RANDOM_INDEX[n] else 0.0
    return dict(weights=dict(zip(CRITERIA[:n], w)), lambda_max=lam, ci=ci, cr=cr)


def topsis(kpi: pd.DataFrame, scenario: str, weights: dict) -> pd.DataFrame:
    """TOPSIS ranking of the four policies in one scenario.
    Criteria: delay (cost), safety score (benefit), CO2 (cost); vector-normalised, weighted,
    closeness to the ideal solution = d(-) / (d(+) + d(-))."""
    g = kpi[kpi.scenario == scenario].groupby("policy")[
        ["avg_delay_s", "safety_score", "co2_kg"]].mean().reindex(POLICIES)
    X = g.values.astype(float)
    R = X / np.sqrt((X ** 2).sum(axis=0))
    W = np.array([weights["Efficiency"], weights["Safety"], weights["Environment"]])
    V = R * W
    benefit = np.array([False, True, False])
    ideal = np.where(benefit, V.max(axis=0), V.min(axis=0))
    anti = np.where(benefit, V.min(axis=0), V.max(axis=0))
    dp = np.sqrt(((V - ideal) ** 2).sum(axis=1))
    dm = np.sqrt(((V - anti) ** 2).sum(axis=1))
    out = pd.DataFrame(dict(policy=POLICIES, closeness=dm / (dp + dm)))
    out["rank"] = out.closeness.rank(ascending=False).astype(int)
    return out


def topsis_all(kpi: pd.DataFrame, weights: dict) -> pd.DataFrame:
    rows = []
    for sc in C.SCENARIO_ORDER:
        t = topsis(kpi, sc, weights)
        rows.append(dict(scenario=sc, **{r.policy: r.closeness for r in t.itertuples()},
                         best=t.loc[t.closeness.idxmax(), "policy"]))
    return pd.DataFrame(rows)
