"""
Smart Traffic Command Centre - Streamlit dashboard.

Multi-agent reinforcement learning (Q-learning, SARSA) for traffic lights, autonomous
vehicles and buses on a 3 x 3 Downtown-Dubai-style grid, evaluated across six scenario
families and scored with a decision network (influence diagram).

Run locally:   streamlit run app.py
All results are precomputed by run_experiments.py into ./data, so the app starts at once.
"""
from __future__ import annotations

import gzip
import json
import os

import numpy as np
import pandas as pd
import streamlit as st

import agents as A
import config as C
import course_methods as M
import decision_network as D
import experiments as X
import lab as LB
import sim_view
import value_of_time as V
import viz

st.set_page_config(page_title="Smart Traffic Command Centre", page_icon="🚦",
                   layout="wide", initial_sidebar_state="expanded")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@500;700&display=swap');
  :root { --bg:#070b14; --panel:#0e1524; --panel2:#121b2e; --line:rgba(148,163,184,0.14);
          --ink:#e8eef8; --muted:#8693a8; --cyan:#22d3ee; --blue:#3d8bff; --amber:#f5b301;
          --green:#21c98b; --red:#ff4d5e; --violet:#a78bfa; }
  html, body, [class*="css"] { font-family: Inter, -apple-system, Segoe UI, Roboto, sans-serif; }
  .stApp { background:
      radial-gradient(1200px 600px at 85% -10%, rgba(61,139,255,0.13), transparent 60%),
      radial-gradient(900px 500px at -10% 10%, rgba(34,211,238,0.08), transparent 60%),
      linear-gradient(rgba(148,163,184,0.035) 1px, transparent 1px) 0 0 / 100% 42px,
      linear-gradient(90deg, rgba(148,163,184,0.035) 1px, transparent 1px) 0 0 / 42px 100%,
      #070b14; }
  .block-container {padding-top: 1.0rem; padding-bottom: 3rem; max-width: 1640px;}
  header[data-testid="stHeader"] {background: transparent;}
  h1, h2, h3, h4 {font-family: 'Space Grotesk', Inter, sans-serif; letter-spacing: -0.01em;}
  /* section headings: accent bar */
  h4 {font-size: 1.22rem !important; font-weight: 600 !important; margin-top: .6rem !important;
      padding-left: 16px !important; position: relative;}
  h4::before {content:""; position:absolute; left:0; top:.28em; bottom:.28em; width:4px;
              border-radius:4px; background:linear-gradient(180deg, var(--cyan), var(--blue));
              box-shadow:0 0 12px rgba(34,211,238,.6);}
  /* metric tiles: glass cards */
  [data-testid="stMetric"] {background: linear-gradient(160deg, rgba(23,33,54,.92), rgba(12,18,31,.92));
      border:1px solid var(--line); border-radius:16px; padding:14px 16px 12px 16px;
      position:relative; overflow:hidden; box-shadow: 0 10px 30px rgba(0,0,0,.25);}
  [data-testid="stMetric"]::after {content:""; position:absolute; inset:0 0 auto 0; height:2px;
      background:linear-gradient(90deg, var(--cyan), rgba(61,139,255,0.0));}
  [data-testid="stMetricValue"] {font-family:'JetBrains Mono', monospace;
      font-size:clamp(1.05rem, 1.75vw, 1.85rem); font-weight:700; color:#f4f8ff;}
  /* never cut a KPI value or label off with "…" in a narrow window (e.g. ~1,100 px) */
  [data-testid="stMetricValue"] > div, [data-testid="stMetricValue"] div,
  [data-testid="stMetricLabel"] div, [data-testid="stMetricLabel"] p {
      overflow:visible !important; text-overflow:clip !important; white-space:normal !important;}
  [data-testid="stMetricLabel"] p {word-break:normal !important; overflow-wrap:normal !important;}
  [data-testid="stMetricDelta"], [data-testid="stMetricDelta"] div {overflow:visible !important;
      text-overflow:clip !important; white-space:nowrap !important;}
  /* narrow windows (about 1,100-1,450 px): a row of 4-6 KPI tiles wraps to rows of three */
  @media (max-width: 1450px) {
      [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] [data-testid="stMetric"]) {
          flex-wrap: wrap !important; row-gap: 0.75rem;}
      [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] [data-testid="stMetric"])
          > [data-testid="stColumn"] {flex: 1 1 calc(33.3% - 1rem) !important;
          min-width: calc(33.3% - 1rem) !important;}
      [data-testid="stMetric"] {padding:10px 12px 8px 12px;}
      [data-testid="stMetricLabel"] p {font-size:.72rem; letter-spacing:.04em;}}
  [data-testid="stMetricLabel"] p {font-size:.80rem; color:var(--muted); text-transform:uppercase;
      letter-spacing:.07em; font-weight:600;}
  [data-testid="stMetricDelta"] {font-size:.88rem; font-weight:600;}
  [data-testid="stCaptionContainer"], .stCaption {font-size:.9rem; color:#8a97ad !important;}
  /* charts and tables on cards */
  [data-testid="stPlotlyChart"] {background: linear-gradient(180deg, rgba(18,27,46,.85), rgba(11,17,29,.85));
      border:1px solid var(--line); border-radius:16px; padding:8px 10px 6px 10px;
      box-sizing: content-box;}
  [data-testid="stDataFrame"] {border:1px solid var(--line); border-radius:12px; overflow:hidden;}
  iframe {border-radius:16px;}
  /* hide the Deploy button and the hamburger menu */
  .stAppDeployButton, [data-testid="stAppDeployButton"], #MainMenu,
  [data-testid="stMainMenu"] {display: none !important;}
  /* navigation: the view radio drawn as a segmented control */
  div[data-testid="stRadio"]:has([aria-label="View"]) [role="radiogroup"] {gap: 6px; flex-wrap: wrap;
      background: rgba(14,21,36,.8); border:1px solid var(--line); border-radius:16px; padding:6px;
      backdrop-filter: blur(8px);}
  div[data-testid="stRadio"]:has([aria-label="View"]) [data-testid="stRadioOption"] {
      border-radius: 11px; padding: 7px 16px; cursor: pointer; margin:0;
      transition: background .15s, box-shadow .15s;}
  div[data-testid="stRadio"]:has([aria-label="View"]) [data-testid="stRadioOption"]:hover {
      background: rgba(61,139,255,.10);}
  div[data-testid="stRadio"]:has([aria-label="View"]) [data-testid="stRadioOption"][data-selected="true"] {
      background: linear-gradient(135deg, #0891b2, #2563eb);
      box-shadow: 0 0 22px rgba(34,211,238,.35), inset 0 0 0 1px rgba(255,255,255,.12);}
  div[data-testid="stRadio"]:has([aria-label="View"]) [data-testid="stRadioOption"] > div > div:first-child {display:none;}
  div[data-testid="stRadio"]:has([aria-label="View"]) [data-testid="stRadioOption"] p {
      font-size: .98rem; font-weight: 600; color: #b9c4d6;}
  div[data-testid="stRadio"]:has([aria-label="View"]) [data-testid="stRadioOption"][data-selected="true"] p {color:#fff;}
  /* inner tabs */
  .stTabs [data-baseweb="tab-list"] {gap: 4px;}
  .stTabs [data-baseweb="tab"] {background: rgba(14,21,36,.7); border:1px solid var(--line);
      border-radius:10px 10px 0 0; padding:6px 14px;}
  /* sidebar */
  section[data-testid="stSidebar"] {background: linear-gradient(180deg, #0b1220, #070b14);
      border-right: 1px solid var(--line);}
  .muted {color:var(--muted); font-size:.95rem;}
  .pill {display:inline-block; padding:4px 12px; border-radius:999px; background:rgba(61,139,255,.10);
         color:#cfe0ff; font-size:.86rem; margin:0 6px 6px 0; border:1px solid rgba(61,139,255,.28);
         font-weight:600;}
  .card {border:1px solid var(--line); border-radius:16px; padding:14px 16px; height:100%;
         background: linear-gradient(160deg, rgba(23,33,54,.9), rgba(12,18,31,.9)); margin-bottom:10px;}
  .card h4 {margin:0 0 6px 0 !important; font-size:1.05rem !important; padding-left:0 !important;}
  .card.sel {border-color: rgba(34,211,238,.6); box-shadow: 0 0 0 1px rgba(34,211,238,.25),
             0 0 28px rgba(34,211,238,.15);}
  .card .tag {float:right; font-size:.7rem; font-weight:700; letter-spacing:.1em; color:#22d3ee;}
  .rules {border:1px solid var(--line); border-radius:12px; overflow:hidden; margin-bottom:12px;}
  .rules div {display:grid; grid-template-columns: 1.1fr 1.1fr 2fr; gap:10px; padding:7px 12px;
              border-top:1px solid var(--line); font-size:.86rem; color:#cbd5e1;}
  .rules div:first-child {border-top:none; background:rgba(61,139,255,.08); color:#8693a8;
              font-size:.74rem; letter-spacing:.08em; text-transform:uppercase; font-weight:700;}
  .rules b {color:#f4f8ff; font-weight:600;}
  .card h4::before {display:none;}
  .card p {margin:0; color:#a9b4c7; font-size:.92rem;}
  /* hero */
  .hero {position:relative; overflow:hidden; border-radius:22px; padding:22px 28px 20px 28px;
         border:1px solid rgba(61,139,255,.25); margin-bottom:14px;
         background:
           radial-gradient(600px 220px at 85% 0%, rgba(34,211,238,.20), transparent 70%),
           radial-gradient(500px 240px at 0% 100%, rgba(61,139,255,.18), transparent 70%),
           linear-gradient(135deg, #0d1730 0%, #0a1222 60%, #0b1020 100%);
         box-shadow: 0 20px 60px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.05);}
  .hero::after {content:""; position:absolute; inset:0; pointer-events:none;
         background: repeating-linear-gradient(90deg, rgba(34,211,238,.05) 0 1px, transparent 1px 60px);
         mask-image: linear-gradient(90deg, transparent, #000 40%, #000 60%, transparent);}
  .hero-grid {display:flex; gap:28px; align-items:flex-end; justify-content:space-between; flex-wrap:wrap;
              position:relative; z-index:1;}
  .eyebrow {font-family:'JetBrains Mono', monospace; font-size:.76rem; letter-spacing:.16em;
            color:var(--cyan); display:flex; align-items:center; gap:10px; text-transform:uppercase;}
  .live {width:9px; height:9px; border-radius:50%; background:#ff4d5e; box-shadow:0 0 0 0 rgba(255,77,94,.7);
         animation: pulse 1.6s infinite;}
  @keyframes pulse {0% {box-shadow:0 0 0 0 rgba(255,77,94,.6);} 70% {box-shadow:0 0 0 12px rgba(255,77,94,0);}
                    100% {box-shadow:0 0 0 0 rgba(255,77,94,0);}}
  .hero h1 {font-size:2.55rem; line-height:1.05; margin:.35rem 0 .35rem 0; padding:0; color:#f4f8ff;
            font-weight:700;}
  .hero h1 .g {background: linear-gradient(90deg, #22d3ee, #3d8bff 55%, #a78bfa);
               -webkit-background-clip:text; background-clip:text; color:transparent;}
  .hero .sub {color:#a9b4c7; font-size:1.0rem; margin:0;}
  .hstats {display:flex; gap:12px; flex-wrap:wrap;}
  .hstat {min-width:140px; padding:10px 14px; border-radius:14px; background:rgba(7,11,20,.55);
          border:1px solid var(--line);}
  .hstat .k {font-size:.70rem; color:var(--muted); letter-spacing:.1em; text-transform:uppercase;
             font-weight:600;}
  .hstat .v {font-family:'JetBrains Mono', monospace; font-size:1.45rem; font-weight:700; color:#f4f8ff;}
  .hstat .v small {font-size:.8rem; color:var(--muted); font-weight:500;}
  .hstat .v.good {color:#34d399;} .hstat .v.bad {color:#ff6b7a;}
  .ctx {margin-top:12px; position:relative; z-index:1;}
  .ctx .desc {color:#8a97ad; font-size:.9rem;}
  /* ticker */
  .ticker {overflow:hidden; white-space:nowrap; border:1px solid var(--line); border-radius:12px;
           background:rgba(10,16,28,.8); margin:0 0 12px 0; position:relative;}
  .ticker::before, .ticker::after {content:""; position:absolute; top:0; bottom:0; width:60px; z-index:2;}
  .ticker::before {left:0; background:linear-gradient(90deg,#0a101c,transparent);}
  .ticker::after {right:0; background:linear-gradient(-90deg,#0a101c,transparent);}
  .ticker .track {display:inline-block; padding:8px 0; animation: scroll 60s linear infinite;}
  .ticker:hover .track {animation-play-state: paused;}
  .ticker .it {display:inline-block; margin:0 26px; font-family:'JetBrains Mono', monospace;
               font-size:.84rem; color:#b9c4d6;}
  .ticker .it b {color:#e8eef8; font-weight:700;}
  .ticker .up {color:#34d399;} .ticker .dn {color:#ff6b7a;}
  .ticker .sep {color:#3d8bff; margin-right:8px;}
  @keyframes scroll {from {transform:translateX(0);} to {transform:translateX(-50%);}}
  /* lab */
  .chip {display:inline-flex; align-items:center; gap:6px; padding:5px 11px; border-radius:999px;
         font-size:.82rem; margin:0 6px 6px 0; border:1px solid var(--line); background:rgba(14,21,36,.8);
         color:#cbd5e1;}
  .chip b {font-family:'JetBrains Mono', monospace; color:#f4f8ff;}
  .chip.link {border-color:rgba(34,211,238,.45); color:#a5f3fc;}
  .verdict {border-radius:16px; padding:14px 18px; border:1px solid rgba(34,211,238,.35);
            background: linear-gradient(135deg, rgba(8,145,178,.18), rgba(37,99,235,.14));}
  .verdict .big {font-family:'Space Grotesk', sans-serif; font-size:1.35rem; font-weight:700; color:#f4f8ff;}
  .verdict .small {color:#a9b4c7; font-size:.9rem;}
  .sbh {font-family:'Space Grotesk', sans-serif; font-weight:700; font-size:1.15rem; color:#f4f8ff;}
  .sbh span {color:var(--cyan);}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ======================================================================================
# Data
# ======================================================================================
@st.cache_data(show_spinner=False)
def load_tables():
    r = lambda n: pd.read_csv(os.path.join(DATA, n))
    out = dict(kpi=r("kpi_results.csv"), curves=r("learning_curves.csv"),
               abl=r("ablation_results.csv"), sim=r("smart_traffic_simulation_log.csv"),
               dec=r("agent_decisions.csv"), dn=r("decision_network_samples.csv"),
               scen=r("scenario_parameters.csv"))
    sel = os.path.join(DATA, "model_selection.csv")
    out["sel"] = pd.read_csv(sel) if os.path.exists(sel) else None
    rob = os.path.join(DATA, "training_robustness.csv")
    out["robust"] = pd.read_csv(rob) if os.path.exists(rob) else None
    ped = os.path.join(DATA, "ped_weight_tradeoff.csv")
    out["ped"] = pd.read_csv(ped) if os.path.exists(ped) else None
    for k, n in [("x_av", "ext_av_speed_lane.csv"), ("x_bus", "ext_bus_load.csv"),
                 ("x_safe", "ext_safety_sensitivity.csv"), ("x_rank", "ext_safety_ranking.csv"),
                 ("x_tune", "ext_actuated_tuning.csv"), ("x_ta", "ext_tuned_actuated.csv"),
                 ("x_guard", "ext_ped_guard.csv"), ("x_sense", "ext_sensing.csv"),
                 ("x_sense_sum", "ext_sensing_summary.csv")]:
        f_ = os.path.join(DATA, n)
        out[k] = pd.read_csv(f_) if os.path.exists(f_) else None
    sim = out["sim"]
    sim["queue_total"] = sim[["queue_n", "queue_e", "queue_s", "queue_w"]].sum(axis=1)
    return out


@st.cache_data(show_spinner=False)
def load_anims():
    with gzip.open(os.path.join(DATA, "animations.json.gz"), "rt") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_models():
    models = X.load_models(os.path.join(DATA, "q_tables.npz"))
    sel = os.path.join(DATA, "model_selection.csv")
    if os.path.exists(sel):                  # deploy the table chosen on validation runs
        X.apply_selection(models, sel)
    return models


@st.cache_resource(show_spinner=False)
def build_dn(_samples: pd.DataFrame, n_rows: int):
    return D.build_traffic_network(_samples)


def show(fig, key=None):
    st.plotly_chart(fig, width="stretch", theme=None, key=key,
                    config={"displaylogo": False, "modeBarButtonsToRemove":
                            ["lasso2d", "select2d", "autoScale2d"]})


COLNAMES = {
    "EU": "Expected utility", "E[U_efficiency]": "Utility: efficiency",
    "E[U_safety]": "Utility: safety", "E[U_environment]": "Utility: environment",
    "P(Congestion=High)": "P(congestion high)", "P(Collision risk=High)": "P(collision risk high)",
    "P(Emissions=High)": "P(emissions high)", "P(Health impact=High)": "P(health impact high)",
    "EVPI": "Value of information (EVPI)", "paired_t_p": "Paired t-test p",
    "q_learning_delay": "Q-learning delay (s)", "sarsa_delay": "SARSA delay (s)",
    "sarsa_minus_q": "SARSA minus Q-learning (s)", "q_learning_conflicts": "Q-learning conflicts / 1,000",
    "sarsa_conflicts": "SARSA conflicts / 1,000", "q_learning_jaywalks": "Q-learning jaywalks",
    "sarsa_jaywalks": "SARSA jaywalks", "conflicts_per_1000": "Conflicts per 1,000 veh",
    "delay_veh_h": "Delay (veh-h)", "t_s": "Time (s)", "time_s": "Time (s)", "q_gap_s": "Q-value gap (s)",
    "share_choosing_x": "Share choosing x-link", "x_link": "x-link", "y_link": "y-link",
}
_TOK = {"avg": "average", "s": "(s)", "kg": "(kg)", "g": "(g)", "vph": "(veh/h)", "pct": "(%)",
        "co2": "CO₂", "nox": "NOx", "ped": "pedestrian", "veh": "veh", "av": "AV", "ns": "NS",
        "ew": "EW", "vs": "vs", "h": "(h)", "id": "ID"}


def nice(col):
    c = str(col)
    if c in COLNAMES:
        return COLNAMES[c]
    if "_" not in c:
        return c[:1].upper() + c[1:] if c[:1].islower() else c
    s = " ".join(_TOK.get(w.lower(), w) for w in c.split("_"))
    return s[:1].upper() + s[1:]


def _digits(series):
    m = float(np.nanmax(np.abs(series.to_numpy(dtype=float)))) if series.notna().any() else 0.0
    return 3 if m < 1 else 2 if m < 10 else 1 if m < 1000 else 0


def pretty(df):
    """Readable column names and sensible rounding for every table on screen."""
    df = df.copy()
    cfg = {}
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            d = _digits(df[c])
            df[c] = df[c].round(d)
            cfg[c] = st.column_config.NumberColumn(nice(c), format=f"%.{d}f")
        elif pd.api.types.is_integer_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
            cfg[c] = st.column_config.NumberColumn(nice(c), format="%d")
        else:
            cfg[c] = st.column_config.Column(nice(c), width="medium")
    return df, cfg


def table(df, height=None, key=None):
    kw = dict(width="stretch", hide_index=True)
    if height:
        kw["height"] = height
    if isinstance(df.index, pd.RangeIndex) is False and df.index.name:
        df = df.reset_index()
    d2, cfg = pretty(df)
    st.dataframe(d2, key=key, column_config=cfg, **kw)


def embed(html: str, height: int):
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:                                            # Streamlit < 1.5x
        import streamlit.components.v1 as components
        components.html(html, height=height, scrolling=False)


# widget values survive a switch of view (widgets of the hidden views are not drawn)
for _k in list(st.session_state.keys()):
    if not _k.startswith("dl-"):
        try:
            st.session_state[_k] = st.session_state[_k]
        except Exception:
            pass

with st.spinner("Loading simulation results ..."):
    T = load_tables()
    ANIMS = load_anims()
    MODELS = load_models()
    DN = build_dn(T["dn"], len(T["dn"]))

KPI = T["kpi"]
LABEL = {k: v["label"] for k, v in C.SCENARIOS.items()}
MEASURE = {"avg_delay_s": "Average delay (s/veh)", "throughput_vph": "Throughput (veh/h)",
           "co2_kg": "CO₂ emitted (kg)", "conflicts_per_1000": "Expected conflicts per 1,000 veh",
           "safety_score": "Safety score (0-100)", "ped_avg_wait_s": "Pedestrian wait (s)",
           "jaywalks": "Jaywalking pedestrians", "bus_on_time_pct": "Bus on time (%)",
           "bus_avg_delay_s": "Bus delay (s)", "switches": "Phase changes per episode",
           "vehicle_delay_s": "Vehicle delay (veh-s)", "queue_total": "Vehicles queued",
           "ped_waiting": "Pedestrians waiting", "expected_conflicts": "Expected conflicts",
           "co2_g": "CO₂ (g)", "departures": "Vehicles discharged", "jaywalks_w": "Jaywalks",
           "nox_g": "NOx emitted (g)", "idle_hours": "Idle time (veh-h)",
           "avg_stops": "Stops per vehicle", "ev_avg_delay_s": "Emergency-vehicle delay (s)",
           "ped_over30_pct": "Pedestrians waiting > 30 s (%)"}
POL = A.POLICY_ORDER

# ======================================================================================
# Sidebar
# ======================================================================================
# linked stakeholder weights: the three weights are shares of one budget, so moving one
# rescales the other two in proportion and the total stays at 100.
W_KEYS = ["w-e", "w-s", "w-v"]
for _k, _v in zip(W_KEYS, (40, 35, 25)):         # = decision_network.DEFAULT_WEIGHTS
    st.session_state.setdefault(_k, _v)
st.session_state.setdefault("w-link", True)


def _rebalance(changed):
    if not st.session_state.get("w-link"):
        return
    new = int(st.session_state[changed])
    others = [k for k in W_KEYS if k != changed]
    rest = 100 - new
    cur = sum(int(st.session_state[k]) for k in others)
    if cur <= 0:
        vals = [rest / 2, rest / 2]
    else:
        vals = [rest * int(st.session_state[k]) / cur for k in others]
    vals = [int(5 * round(v / 5)) for v in vals]
    vals[int(np.argmax(vals))] += rest - sum(vals)            # keep the total at exactly 100
    for k, v in zip(others, vals):
        st.session_state[k] = int(min(100, max(0, v)))


def _link_toggled():
    if st.session_state.get("w-link"):
        tot = sum(int(st.session_state[k]) for k in W_KEYS) or 1
        vals = [int(5 * round(100 * int(st.session_state[k]) / tot / 5)) for k in W_KEYS]
        if sum(vals) == 0:
            vals = [40, 35, 25]
        vals[int(np.argmax(vals))] += 100 - sum(vals)
        for k, v in zip(W_KEYS, vals):
            st.session_state[k] = v


with st.sidebar:
    st.markdown('<div class="sbh">🚦 Traffic <span>Command</span> Centre</div>', unsafe_allow_html=True)
    st.markdown('<span class="muted">Multi-agent Q-learning & SARSA with decision-network '
                'evaluation · 3 × 3 Downtown-Dubai-style grid</span>', unsafe_allow_html=True)
    scenario = st.selectbox("Scenario", C.SCENARIO_ORDER, format_func=lambda s: LABEL[s],
                            index=1)
    policy = st.selectbox("Control policy", POL, index=2)
    st.markdown("**Stakeholder weights** (multi-attribute utility)")
    st.toggle("Link weights (always sum to 100)", key="w-link", on_change=_link_toggled,
              help="The weights are shares of one budget: raising one lowers the other two "
                   "in proportion. Switch off to set them freely (they are then normalised).")
    w_e = st.slider("Efficiency", 0, 100, step=5, key="w-e", on_change=_rebalance, args=("w-e",))
    w_s = st.slider("Safety", 0, 100, step=5, key="w-s", on_change=_rebalance, args=("w-s",))
    w_v = st.slider("Environment", 0, 100, step=5, key="w-v", on_change=_rebalance, args=("w-v",))
    if w_e + w_s + w_v == 0:
        st.warning("All weights are zero; using the defaults 40 / 35 / 25.")
        w_e, w_s, w_v = 40, 35, 25
    tot = w_e + w_s + w_v
    WEIGHTS = {"U_efficiency": w_e / tot, "U_safety": w_s / tot, "U_environment": w_v / tot}
    st.caption(f"Normalised: efficiency {WEIGHTS['U_efficiency']:.2f} · safety "
               f"{WEIGHTS['U_safety']:.2f} · environment {WEIGHTS['U_environment']:.2f}")
    with st.expander("Download the data"):
        for name in ["smart_traffic_simulation_log.csv", "kpi_results.csv",
                     "decision_network_samples.csv", "learning_curves.csv",
                     "ablation_results.csv", "agent_decisions.csv", "training_robustness.csv",
                     "ped_weight_tradeoff.csv", "scenario_parameters.csv"]:
            if not os.path.exists(os.path.join(DATA, name)):
                continue
            with open(os.path.join(DATA, name), "rb") as f:
                st.download_button(name, f.read(), file_name=name, mime="text/csv",
                                   key=f"dl-{name}")
    st.caption("Each episode = 30 simulated minutes; KPIs exclude the first 5 minutes "
               "(warm-up). Evaluation uses 8 held-out seeds with identical arrivals for "
               "every policy.")
    st.markdown("**Project links**  \n"
                "[Colab notebook](https://colab.research.google.com/github/Sbot-c/Smart_Traffic_U/"
                "blob/main/Smart_Traffic_MARL_DecisionNetwork.ipynb) · "
                "[Code and data (GitHub)](https://github.com/Sbot-c/Smart_Traffic_U)")
    st.caption("Everything here is a live simulation: a replay of simulated data, not live "
               "city sensor feeds.")

SC = C.SCENARIOS[scenario]
k_sel = KPI[(KPI.scenario == scenario) & (KPI.policy == policy)]
k_base = KPI[(KPI.scenario == scenario) & (KPI.policy == "Fixed-time")]


def _hstat(label, value, unit="", cls=""):
    return (f'<div class="hstat"><div class="k">{label}</div>'
            f'<div class="v {cls}">{value}<small> {unit}</small></div></div>')


_d, _db = k_sel.avg_delay_s.mean(), k_base.avg_delay_s.mean()
_c, _cb = k_sel.co2_kg.mean(), k_base.co2_kg.mean()
_chg = 100 * (_d - _db) / _db if _db else 0.0
_co2chg = 100 * (_c - _cb) / _cb if _cb else 0.0
_wx = {"Clear": "☀ Clear", "Rain": "🌧 Rain", "Sandstorm": "🌪 Sandstorm"}[SC["weather"]]
st.markdown(
    '<div class="hero"><div class="hero-grid"><div>'
    '<div class="eyebrow"><span class="live"></span>Live simulation · replay of simulated '
    'data · adaptive signal control · '
    '9 junctions · 36 approaches</div>'
    '<h1>Smart Traffic <span class="g">Command Centre</span></h1>'
    '<p class="sub">Traffic lights, autonomous vehicles and buses learning together with '
    'Q-learning & SARSA, judged by a decision network on efficiency, safety and emissions.</p>'
    '</div><div class="hstats">'
    + _hstat("Avg delay", f"{_d:.1f}", "s/veh")
    + ("" if policy == "Fixed-time" else
       _hstat("vs fixed-time", f"{_chg:+.0f}%", "delay", "good" if _chg < 0 else "bad"))
    + _hstat("Throughput", f"{k_sel.throughput_vph.mean():,.0f}", "veh/h")
    + ("" if policy == "Fixed-time" else
       _hstat("CO₂ change", f"{_co2chg:+.0f}%", "", "good" if _co2chg < 0 else "bad"))
    + '</div></div>'
    f'<div class="ctx"><span class="pill">{SC["label"]}</span><span class="pill">{policy}</span>'
    f'<span class="pill">{_wx}</span><span class="desc">{SC["description"]}</span></div></div>',
    unsafe_allow_html=True)

# ticker: every scenario, best package against the fixed-time plan (delay)
_g = KPI.groupby(["scenario", "policy"]).avg_delay_s.mean().unstack()
_items = []
for _s in C.SCENARIO_ORDER:
    _row = _g.loc[_s]
    _best = _row.drop("Fixed-time").idxmin()
    _v = 100 * (_row[_best] - _row["Fixed-time"]) / _row["Fixed-time"]
    _items.append(f'<span class="it"><span class="sep">▍</span>{LABEL[_s].upper()} · best package '
                  f'<b>{_best}</b> {_row[_best]:.1f} s/veh '
                  f'<span class="{"up" if _v < 0 else "dn"}">{_v:+.0f}% delay</span> vs fixed-time</span>')
_sx0 = T["abl"][T["abl"].study == "Signal x routing"]
for _s, _nm in (("rush_hour", "RUSH HOUR"), ("event_surge", "EVENT SURGE")):
    _q = _sx0[(_sx0.scenario == _s) & (_sx0.variant == "Q-learning + Cooperative RL")].avg_delay_s
    _a = _sx0[(_sx0.scenario == _s) & (_sx0.variant == "Actuated + Cooperative RL")].avg_delay_s
    if len(_q) and len(_a):
        _v = 100 * (_q.mean() - _a.mean()) / _a.mean()
        _items.append(f'<span class="it"><span class="sep">▍</span>SAME ROUTING, {_nm} · '
                      f'Q-learning vs actuated signals <b>{_q.mean():.1f} vs {_a.mean():.1f} s</b> '
                      f'<span class="{"up" if _v < 0 else "dn"}">{_v:+.0f}%</span></span>')
if T.get("x_ta") is not None:
    _ta = T["x_ta"][T["x_ta"].routing == "Cooperative RL"].groupby(["scenario", "controller"]) \
        .avg_delay_s.mean()
    for _s, _nm in (("rush_hour", "RUSH HOUR"), ("event_surge", "EVENT SURGE")):
        _q, _a = _ta[(_s, "Q-learning (deployed)")], _ta[(_s, "Actuated (tuned)")]
        _items.append(f'<span class="it"><span class="sep">▍</span>VS TUNED ACTUATED, {_nm} · '
                      f'Q-learning <b>{_q:.1f}</b> vs <b>{_a:.1f} s</b> '
                      f'<span class="{"up" if _q < _a else "dn"}">{100 * (_q - _a) / _a:+.0f}%</span></span>')
_track = "".join(_items)
st.markdown(f'<div class="ticker"><div class="track">{_track}{_track}</div></div>',
            unsafe_allow_html=True)

VIEWS = ["City Overview", "RL Simulation", "Decision Network Visualizer",
         "Scenario Explorer", "KPI & Trade-off Analysis", "Policy Comparison", "Simulation Lab",
         "Research Extensions"]
ICONS = {"City Overview": "◉", "RL Simulation": "⟳", "Decision Network Visualizer": "◇",
         "Scenario Explorer": "☰", "KPI & Trade-off Analysis": "◆",
         "Policy Comparison": "⇄", "Simulation Lab": "⚡", "Research Extensions": "✚"}
if st.session_state.get("view") == "Live What-If":          # old name in a saved session
    st.session_state["view"] = "Simulation Lab"
# Only the open view is drawn (st.tabs would build all seven on every click).
view = st.radio("View", VIEWS, horizontal=True, label_visibility="collapsed", key="view",
                format_func=lambda v: f"{ICONS[v]}  {v}")

MINUTE_HELP = ("Moves the replay to this minute and sets the network snapshot. The animated "
               "replay holds the first 15 minutes; later minutes park it at 14:30.")


def minute_slider():
    return st.slider("Minute of the episode", 0, 29, 12, key="minute", help=MINUTE_HELP)


minute = 12


def anim_payload(sc, pol):
    p = dict(ANIMS[f"{sc}|{pol}"])
    p["bus_lane"] = bool(C.SCENARIOS[sc].get("bus_lane"))
    p["event"] = bool(C.SCENARIOS[sc].get("event"))
    return p


def mean(df, col):
    return float(df[col].mean()) if len(df) else float("nan")


def pct(v, b):
    return 100.0 * (v - b) / b if b else 0.0


# ======================================================================================
# 1. City overview
# ======================================================================================
if view == VIEWS[0]:
    cols = st.columns(6)
    tiles = [("Avg delay", "avg_delay_s", "s/veh", True, "{:.1f}"),
             ("Throughput", "throughput_vph", "veh/h", False, "{:,.0f}"),
             ("CO₂ emitted", "co2_kg", "kg", True, "{:,.0f}"),
             ("Safety score", "safety_score", "/100", False, "{:.1f}"),
             ("Bus on time", "bus_on_time_pct", "%", False, "{:.0f}"),
             ("Pedestrian wait", "ped_avg_wait_s", "s", True, "{:.1f}")]
    for c, (lab, col, unit, lower_better, fmt) in zip(cols, tiles):
        v, b = mean(k_sel, col), mean(k_base, col)
        delta = None if policy == "Fixed-time" else f"{pct(v, b):+.1f}% vs fixed"
        c.metric(f"{lab} ({unit})", fmt.format(v), delta,
                 delta_color="inverse" if lower_better else "normal")
    st.caption("Policies are packages: *Actuated* = actuated signals with fixed routes; "
               "*Q-learning* / *SARSA* = RL signals with cooperative AV routing. "
               "Signal-versus-routing effects are separated in Policy Comparison. "
               "Means over 8 evaluation episodes; deltas compare against the Webster "
               "fixed-time plan with the same arrivals.")

    minute = minute_slider()
    left, right = st.columns([1.25, 1])
    with left:
        st.markdown("#### Real-time traffic flow")
        st.caption("Replay of evaluation episode 1 (first 15 minutes). Roads are coloured "
                   "by queue per lane (green = free, red = long queue); counters accumulate "
                   "from 00:00. Press space to play or pause.")
        embed(sim_view.render_html([anim_payload(scenario, policy)], [policy],
                                   start=min(minute * 60, 870), autoplay=True), height=780)
    with right:
        sim = T["sim"]
        sim_sc = sim[(sim.scenario == scenario) & (sim.policy == policy)]
        snap = sim_sc[sim_sc.t_start_s == minute * 60]
        q = [[0, 0, 0, 0] for _ in range(C.N_INT)]
        ph = [0] * C.N_INT
        for _, r in snap.iterrows():
            i = C.JUNCTION_NAMES.index(r.junction)
            q[i] = [r.queue_n, r.queue_e, r.queue_s, r.queue_w]
            ph[i] = 0 if r.ns_green_share >= 0.5 else 1
        show(viz.network_map(q, ph, incident=SC.get("incident") if SC.get("incident") and
                             SC["incident"]["t_start"] <= minute * 60 < SC["incident"]["t_end"]
                             else None, height=430), key="map")
        st.caption("Bubble size and colour = mean queue on each approach in that 30-second "
                   "window; badge = phase that held green most of the window.")
        jt = sim_sc.groupby("junction").agg(
            delay_veh_h=("vehicle_delay_s", lambda x: x.sum() / 3600),
            expected_conflicts=("expected_conflicts", "sum"),
            co2_kg=("co2_g", lambda x: x.sum() / 1000), jaywalks=("jaywalks", "sum"),
            phase_changes=("phase_changes", "sum")).reset_index()
        table(jt.round(2), height=360)

    st.markdown("#### Network queue over the episode")
    show(viz.timeseries(T["sim"], scenario, "queue_total",
                        "Vehicles queued (all junctions)", height=300, agg="sum"),
         key="ts-queue")
    st.caption("Mean queue in each 30-second window of evaluation episode 1, summed over "
               "the nine junctions.")

# ======================================================================================
# 2. RL simulation
# ======================================================================================
if view == VIEWS[1]:
    c1, c2, c3 = st.columns([1, 1, 1.6])
    left_pol = c1.selectbox("Left view", POL, index=0, key="cmp-left")
    right_pol = c2.selectbox("Right view", POL, index=POL.index(policy) if policy != "Fixed-time"
                             else 2, key="cmp-right")
    with c3:
        minute = minute_slider()
    embed(sim_view.render_html([anim_payload(scenario, left_pol),
                                anim_payload(scenario, right_pol)],
                               [left_pol, right_pol], start=min(minute * 60, 870)), height=740)
    st.caption("Same arrivals, same seed, one clock: only the control policy differs. Counters "
               "accumulate from 00:00 (cars queued now, queue delay, CO₂); the right-hand tiles "
               "show the change against the left view. Space = play/pause.")

    st.markdown("#### Agent decisions")
    d1, d2, d3 = st.tabs(["Traffic-light agents", "Bus agents", "Autonomous-vehicle routing"])
    with d1:
        jn = st.selectbox("Junction", C.JUNCTION_NAMES, index=4, key="trace-j")
        i = C.JUNCTION_NAMES.index(jn)
        P = ANIMS[f"{scenario}|{policy}"]
        rows = []
        for s in range(len(P["phase"])):
            ov = P["override"][s][i]
            ch = P["chosen"][s][i]
            sw = P["switched"][s][i]
            if sw or ov or ch:
                rows.append(dict(time=f"{int(s * 5 // 60):02d}:{int(s * 5 % 60):02d}",
                                 phase_after=C.PHASE_NAMES[P["phase"][s][i]],
                                 agent_chose="change" if ch else "keep",
                                 executed="change" if sw else "keep",
                                 shield=X.TE.OVERRIDE_NAMES[ov] if ov else "",
                                 queue_N=P["queues"][s][i][0], queue_E=P["queues"][s][i][1],
                                 queue_S=P["queues"][s][i][2], queue_W=P["queues"][s][i][3]))
        st.caption(f"{policy} at {jn}: every 5-second step where the agent asked to change "
                   "phase, the phase changed, or the safety shield intervened (first 15 min).")
        table(pd.DataFrame(rows), height=320)
    with d2:
        dec = T["dec"]
        bd = dec[(dec.scenario == scenario) & (dec.policy == policy) & (dec.agent == "bus")]
        if len(bd):
            counts = bd.action.value_counts().rename_axis("action").reset_index(name="decisions")
            cc1, cc2 = st.columns([1, 2])
            with cc1:
                table(counts)
            with cc2:
                table(bd[["t_s", "agent_id", "junction", "action", "detail", "state"]]
                      .rename(columns={"t_s": "time_s"}), height=300)
        else:
            st.info("No bus decisions recorded for this run.")
    with d3:
        dec = T["dec"]
        ad = dec[(dec.scenario == scenario) & (dec.policy == policy) & (dec.agent == "AV")]
        if len(ad):
            st_ = ad.state.apply(json.loads)
            ad = ad.assign(x_link=st_.apply(lambda d: d["x_link"]),
                           y_link=st_.apply(lambda d: d["y_link"]))
            piv = ad.assign(chose_x=(ad.detail == "x-move").astype(int)).groupby(
                ["x_link", "y_link"]).agg(decisions=("chose_x", "size"),
                                          share_choosing_x=("chose_x", "mean")).reset_index()
            st.caption("Each AV with two shortest-path options picks the next link from the "
                       "congestion on both candidates. Share choosing the x-direction link:")
            table(piv.round(2))
        else:
            st.info("Static routing: autonomous vehicles follow fixed shortest paths under "
                    "this policy.")

    if policy in X.ALGOS:
        st.markdown("#### Learned signal policy")
        m = X.policy_tables(MODELS, policy, scenario)["signals"]
        c = st.columns(5)
        jn2 = c[0].selectbox("Junction", C.JUNCTION_NAMES, index=4, key="q-j")
        phs = c[1].selectbox("Phase", ["NS green", "EW green"], key="q-ph")
        gb = c[2].selectbox("Green so far", A.GREEN_BINS[1:], index=1, key="q-g")
        ab = c[3].selectbox("Arrivals on green (next 6 s)", ["none", "1-2", "3+"], key="q-a")
        fl = c[4].selectbox("Flag", ["none", "pedestrians waiting", "spillback"], key="q-f")
        j = C.JUNCTION_NAMES.index(jn2)
        Q, N = m[j]["Q"], m[j]["visits"]
        ph_i = 0 if phs.startswith("NS") else 1
        g_i = A.GREEN_BINS.index(gb)
        a_i = ["none", "1-2", "3+"].index(ab)
        f_i = ["none", "pedestrians waiting", "spillback"].index(fl)
        grid = np.full((A.N_Q, A.N_Q), np.nan)
        for qg in range(A.N_Q):
            for qr in range(A.N_Q):
                s = A.encode_signal_state(ph_i, g_i, qg, qr, a_i, f_i)
                tried = [a for a in (0, 1) if N[s, a] > 0]
                if tried:
                    grid[qg, qr] = float(max(tried, key=lambda a: Q[s, a]))
        show(viz.policy_heatmap(grid, A.QUEUE_BINS, A.QUEUE_BINS, height=360,
                                title=f"{policy} greedy action at {jn2} ({LABEL[scenario]})"),
             key="qheat")
        st.caption("Cells show the greedy action learned for each queue combination; '–' = "
                   "state never visited in training (the agent falls back to the actuated "
                   "rule there).")
        cb1, cb2 = st.columns(2)
        with cb1:
            st.markdown("**Bus agent policy (most visited states)**")
            bq = X.policy_tables(MODELS, policy, scenario)["bus"]
            vis = bq["visits"].sum(1)
            rows = []
            for s in np.argsort(-vis)[:12]:
                if vis[s] == 0:
                    continue
                dd = A.decode_bus_state(int(s))
                tried = [a for a in range(3) if bq["visits"][s, a] > 0]
                best = max(tried, key=lambda a: bq["Q"][s, a])
                rows.append(dict(**dd, visits=int(vis[s]),
                                 best=["go", "request priority", "hold 20 s"][best]))
            table(pd.DataFrame(rows), height=330)
        with cb2:
            st.markdown("**AV routing policy (shared fleet table)**")
            aq = X.policy_tables(MODELS, policy, scenario)["routing"]
            lv = ["free", "moderate", "heavy", "incident"]
            rows = []
            for cx in range(4):
                for cy in range(4):
                    qx = qy = 0.0
                    n = 0
                    for mx in (1, 2):
                        for my in (1, 2):
                            s = A.av_state(mx, my, cx, cy)
                            if aq["visits"][s].sum() > 0:
                                qx += aq["Q"][s, 0]
                                qy += aq["Q"][s, 1]
                                n += int(aq["visits"][s].sum())
                    if n:
                        rows.append(dict(x_link=lv[cx], y_link=lv[cy], visits=n,
                                         prefers="x-move" if qx >= qy else "y-move",
                                         q_gap_s=round(abs(qx - qy) * 10 / 4, 1)))
            table(pd.DataFrame(rows), height=330)

    st.markdown("#### Learning curves")
    st.caption("Every training episode samples its own pedestrian demand, weather and demand "
               "level (domain randomisation), so the per-episode delay moves with the sampled "
               "conditions as well as with learning. Held-out evaluation is in Policy "
               "Comparison. In rush hour the adaptation curve rises (about 72 to 82 s for "
               "Q-learning) even after allowing for weather and pedestrian levels: extra "
               "rush-hour training made the policy worse, so the general table is deployed "
               "there (chosen on validation runs, see Policy Comparison).")
    cl1, cl2 = st.columns(2)
    with cl1:
        show(viz.learning_curves(T["curves"], "avg_delay_s", "pre-train", height=320,
                                 title="Pre-training on all scenarios (shared table)"),
             key="lc1")
    with cl2:
        show(viz.learning_curves(T["curves"], "avg_delay_s", "adapt", scenario=scenario,
                                 window=8, height=320,
                                 title=f"Adaptation to {LABEL[scenario]}"), key="lc2")

# ======================================================================================
# 3. Decision network
# ======================================================================================
if view == VIEWS[2]:
    st.markdown("#### Influence diagram: which control strategy maximises expected utility?")
    c = st.columns([1, 1, 1, 1.3])
    use_sc = c[3].toggle("Use the selected scenario's conditions", value=True)
    default = D.SCENARIO_EVIDENCE.get(scenario, {}) if use_sc else {}
    opts = lambda st_: ["unknown"] + st_
    ev = {}
    for col, node, states in [(c[0], "Weather", D.WEATHER),
                              (c[1], "Traffic density", D.DENSITY),
                              (c[2], "Pedestrian flow", D.PED)]:
        val = col.selectbox(node, opts(states),
                            index=opts(states).index(default.get(node, "unknown")),
                            key=f"ev-{node}-{use_sc}-{scenario}")
        if val != "unknown":
            ev[node] = val
    best, tab_eu = DN.best(ev, WEIGHTS)
    c2 = st.columns([1, 1, 2])
    dsig = c2[0].selectbox("Inspect signal timing", ["best"] + D.SIGNAL, key="dn-sig")
    drou = c2[1].selectbox("Inspect routing", ["best"] + D.ROUTING, key="dn-rou")
    decision = {"Signal timing": best["Signal timing"] if dsig == "best" else dsig,
                "Routing": best["Routing"] if drou == "best" else drou}
    res = DN.expected_utility(decision, ev, WEIGHTS)
    c2[2].markdown(f"**Recommended:** {best['Signal timing']} signals + {best['Routing']} "
                   f"routing &nbsp;·&nbsp; expected utility **{best['EU']:.1f}**/100  \n"
                   f'<span class="muted">Evidence: '
                   f'{", ".join(f"{k} = {v}" for k, v in ev.items()) or "none (priors)"}</span>',
                   unsafe_allow_html=True)
    left, right = st.columns([1.35, 1])
    with left:
        show(viz.influence_diagram(DN, res["posteriors"], decision, res["components"],
                                   height=560), key="dn-diag")
        st.caption("Ovals = chance nodes (blue), rectangles = decisions (orange), diamonds = "
                   "utilities (green). Each chance node shows its most likely state given the "
                   "evidence and the inspected decision; hover for the full distribution.")
    with right:
        show(viz.eu_bars(tab_eu, height=330, title="Expected utility of every decision pair"),
             key="dn-eu")
        show(viz.posterior_bars(DN, res["posteriors"], ["Congestion", "Collision risk",
                                                        "Emissions"], height=230,
                                title="Outcome probabilities for the inspected decision"),
             key="dn-post")
    st.markdown("#### Expected-utility table")
    show_cols = ["Signal timing", "Routing", "EU", "E[U_efficiency]", "E[U_safety]",
                 "E[U_environment]", "P(Congestion=High)", "P(Collision risk=High)",
                 "P(Emissions=High)", "P(Health impact=High)"]
    table(tab_eu[show_cols].round(3))
    cv1, cv2 = st.columns([1, 1.2])
    with cv1:
        st.markdown("#### Value of information")
        rows = []
        voi_ev = ev if len(ev) < 3 else {}
        if len(ev) == 3:
            st.caption("All three conditions are set above, so this table values observing "
                       "each one starting from the priors (no evidence).")
        for v in ["Weather", "Traffic density", "Pedestrian flow"]:
            if v in voi_ev:
                continue
            r = DN.value_of_information(v, voi_ev, WEIGHTS)
            rows.append(dict(observe=v, EVPI=round(r["evpi"], 3),
                             best_if_known="; ".join(f"{d['state']}: {d['best_signal']} + "
                                                     f"{d['best_routing']}"
                                                     for d in r["detail"])))
        table(pd.DataFrame(rows))
        st.caption("EVPI = expected gain in utility from observing the variable before "
                   "choosing. Zero means the same decision is best whatever its value.")
        if rows and all(r["EVPI"] < 0.005 for r in rows):   # < 0.005 utility points = nil
            st.info("Reading: observing any of these conditions would not change the best "
                    "decision, so extra information is worth nothing here. The choice is "
                    "robust to what the sensors report.")
        elif rows:
            top = max(rows, key=lambda r: r["EVPI"])
            st.info(f"Reading: knowing **{top['observe']}** is worth the most "
                    f"(EVPI {top['EVPI']:.2f} utility points). It is the condition worth "
                    f"sensing before the signal and routing plan is chosen.")
        st.markdown("#### Discretisation")
        e = DN.edges
        st.markdown(f"- **Congestion** (control delay per junction): Low ≤ {e['congestion'][0]:.0f} s "
                    f"(LOS A), Medium ≤ {e['congestion'][1]:.0f} s (LOS B), High > "
                    f"{e['congestion'][1]:.0f} s (LOS C or worse)\n"
                    f"- **Collision risk** (expected conflicts per 1,000 vehicles): tertiles at "
                    f"{e['risk'][0]:.2f} and {e['risk'][1]:.2f}\n"
                    f"- **Emissions** (g CO₂ from idling and stops per vehicle): tertiles at "
                    f"{e['emissions'][0]:.1f} and {e['emissions'][1]:.1f}\n"
                    f"- Learned from **{len(DN.data):,}** junction × 10-minute samples "
                    f"(216 condition × decision combinations × 2 seeds).")
    with cv2:
        st.markdown("#### Sensitivity to stakeholder weights")
        ws = D.weight_sensitivity(DN, ev, step=0.05)
        show(viz.ternary_map(ws, height=440), key="dn-tern")
        _share = ws.best_signal.value_counts(normalize=True)
        if len(_share) == 1:
            st.info(f"Reading: **{_share.index[0]}** signals are best for every weighting "
                    "tried, so the recommendation does not depend on how stakeholders "
                    "trade efficiency, safety and environment.")
        else:
            st.info("Reading: " + ", ".join(f"**{p}** is best for {100 * v:.0f}% of the "
                                            "weightings" for p, v in _share.items())
                    + ". The recommendation flips where the colours meet.")
        st.caption("Each dot is one combination of efficiency, safety and environment "
                   "weights (steps of 0.05); its colour is the signal policy with the highest "
                   "expected utility under the evidence above.")
    st.markdown("#### Conditional probability tables")
    node = st.selectbox("Node", DN.order, index=DN.order.index("Congestion"), key="cpt-node")
    cpt = D.cpt_frame(DN)
    table(cpt[cpt.node == node].drop(columns=["node"]).dropna(axis=1, how="all"), height=320)

# ======================================================================================
# 4. Scenario explorer
# ======================================================================================
if view == VIEWS[3]:
    st.markdown("#### Scenarios")
    scen = T["scen"]
    cols = st.columns(4)
    for k, key in enumerate(C.SCENARIO_ORDER):
        r = scen[scen.scenario == key].iloc[0]
        cols[k % 4].markdown(
            f'<div class="card{" sel" if key == scenario else ""}">'
            f'{"<span class=tag>SELECTED</span>" if key == scenario else ""}'
            f'<h4>{r.label}</h4><p>{r.description}</p>'
            f'<p style="margin-top:6px"><span class="pill">{r.weather}</span>'
            f'<span class="pill">demand ×{r.demand_multiplier:g}</span>'
            f'<span class="pill">bus every {int(r.bus_headway_s // 60)} min</span></p></div>',
            unsafe_allow_html=True)
        if k % 4 == 3:
            cols = st.columns(4)
    st.markdown(f"#### {LABEL[scenario]}: outcome distributions across 8 evaluation episodes")
    ks = KPI[KPI.scenario == scenario]
    b1, b2, b3 = st.columns(3)
    with b1:
        show(viz.box_by_policy(ks, "avg_delay_s", "s/veh", height=300,
                               title="Average delay"), key="bx1")
    with b2:
        show(viz.box_by_policy(ks, "conflicts_per_1000", "per 1,000 veh", height=300,
                               title="Expected conflicts"), key="bx2")
    with b3:
        show(viz.box_by_policy(ks, "co2_kg", "kg", height=300, title="CO₂ emitted"),
             key="bx3")
    c1, c2 = st.columns([1, 1])
    jsel = c1.selectbox("Junction", ["All junctions"] + C.JUNCTION_NAMES, key="ts-j")
    msel = c2.selectbox("Measure", ["vehicle_delay_s", "queue_total", "ped_waiting",
                                    "expected_conflicts", "co2_g", "departures", "jaywalks"],
                        format_func=lambda m: MEASURE.get(m, m), key="ts-m")
    show(viz.timeseries(T["sim"], scenario, msel, MEASURE.get(msel, msel), junction=jsel,
                        height=320, title=f"{MEASURE.get(msel, msel)} per 30-second window"),
         key="ts2")
    st.markdown("#### What the decision network recommends here")
    evs = D.SCENARIO_EVIDENCE[scenario]
    _, t = DN.best(evs, WEIGHTS)
    sim_rank = ks.groupby("policy")[["avg_delay_s", "conflicts_per_1000", "co2_kg",
                                     "ped_avg_wait_s"]].mean().round(2).sort_values(
        "avg_delay_s")
    cc1, cc2 = st.columns(2)
    with cc1:
        st.caption("Decision network (evidence: " + ", ".join(f"{k} = {v}" for k, v in
                                                              evs.items()) + ")")
        table(t[["Signal timing", "Routing", "EU", "E[U_efficiency]", "E[U_safety]",
                 "E[U_environment]"]].head(4).round(2))
    with cc2:
        sx = T["abl"][(T["abl"].study == "Signal x routing") & (T["abl"].scenario == scenario)]
        if len(sx):
            st.caption("Direct simulation of the same signal × routing pairs (8 episodes)")
            table(sx.groupby("variant")[["avg_delay_s", "conflicts_per_1000", "co2_kg"]]
                  .mean().round(2).sort_values("avg_delay_s").reset_index().head(4))
        else:
            st.caption("Direct simulation of the four integrated policies (8 episodes)")
            table(sim_rank.reset_index())

# ======================================================================================
# 5. KPI & trade-off analysis
# ======================================================================================
if view == VIEWS[4]:
    st.markdown("#### Improvement over the fixed-time plan")
    h1, h2 = st.columns(2)
    with h1:
        show(viz.improvement_heatmap(KPI, "avg_delay_s", title="Average delay", height=330),
             key="hm1")
    with h2:
        show(viz.improvement_heatmap(KPI, "conflicts_per_1000", title="Expected conflicts",
                                     height=330), key="hm3")
    h3, h4 = st.columns(2)
    with h3:
        show(viz.improvement_heatmap(KPI, "co2_kg", title="CO₂ emissions", height=330),
             key="hm2")
    with h4:
        show(viz.improvement_heatmap(KPI, "ped_avg_wait_s", title="Pedestrian waiting time",
                                     height=330), key="hm4")
    h5, h6 = st.columns(2)
    with h5:
        show(viz.improvement_heatmap(KPI, "bus_on_time_pct", higher_is_better=True,
                                     title="Bus on-time performance", height=330), key="hm5")
    with h6:
        show(viz.improvement_heatmap(KPI, "nox_g", title="NOx emissions", height=330), key="hm6")
    h7, h8 = st.columns(2)
    with h7:
        show(viz.improvement_heatmap(KPI, "idle_hours", title="Idle time (engine running, "
                                     "stationary)", height=330), key="hm7")
    with h8:
        show(viz.improvement_heatmap(KPI, "throughput_vph", higher_is_better=True,
                                     title="Throughput", height=330), key="hm8")
    st.caption("Blue = better than fixed-time, red = worse; labels give the raw % change. "
               "Each row is a package (signals + routing), so a gap between policies mixes a "
               "signal effect and a routing effect - see Policy Comparison. \"Safety\" is a "
               "stand-in built from expected conflicts, not observed crashes. The RL agents "
               "had 280 pre-training episodes plus 80 adaptation episodes per scenario; the "
               "actuated rule here is untuned (see Research Extensions for the tuned one).")
    st.markdown("#### Trade-offs")
    t1, t2 = st.columns([1.2, 1])
    with t1:
        fig, tt = viz.tradeoff_scatter(KPI, height=430,
                                       title="Delay vs safety, all scenarios (bubble = CO₂ saved)")
        show(fig, key="tradeoff")
    with t2:
        st.markdown("**Weighted-sum score** (criteria min-max normalised within each "
                    "scenario; weights from the sidebar)")
        g = KPI.groupby(["scenario", "policy"])[["avg_delay_s", "safety_score", "co2_kg"]] \
            .mean().reset_index()
        out = []
        for s_, sub in g.groupby("scenario"):
            sub = sub.copy()
            for col, better_low in [("avg_delay_s", True), ("safety_score", False),
                                    ("co2_kg", True)]:
                lo, hi = sub[col].min(), sub[col].max()
                rng = (hi - lo) or 1.0
                sub[f"u_{col}"] = 100 * ((hi - sub[col]) / rng if better_low
                                         else (sub[col] - lo) / rng)
            sub["score"] = (WEIGHTS["U_efficiency"] * sub.u_avg_delay_s + WEIGHTS["U_safety"]
                            * sub.u_safety_score + WEIGHTS["U_environment"] * sub.u_co2_kg)
            out.append(sub)
        ws_ = pd.concat(out)
        piv = ws_.pivot(index="scenario", columns="policy", values="score").reindex(
            C.SCENARIO_ORDER)[POL]
        piv.index = [LABEL[s] for s in piv.index]
        def _best(row):
            top = row.max()
            near = [p for p in POL if row[p] >= top - 0.5]       # within half a point = a tie
            return near[0] if len(near) == 1 else "Tie: " + " / ".join(near)
        piv["best"] = piv[POL].apply(_best, axis=1)
        piv = piv.rename(columns={"best": "Best policy"})
        table(piv.round(1).rename_axis("Scenario").reset_index())
        st.caption("Scores within 0.5 points of the top are shown as a tie.")
        sel = ws_[ws_.scenario == scenario].set_index("policy")
        rad = pd.DataFrame({"Efficiency": sel.u_avg_delay_s, "Safety": sel.u_safety_score,
                            "Environment": sel.u_co2_kg})
        show(viz.radar(rad, height=330, title=f"Normalised criteria - {LABEL[scenario]}"),
             key="radar")
    with st.expander("Table view of the trade-off chart"):
        table(tt.assign(scenario=tt.scenario.map(LABEL)).round(2))
    st.markdown("#### Decision-science toolkit (course techniques)")
    k1, k2, k3, k4 = st.tabs(["Uncertainty criteria", "Risk-averse utility",
                              "Nash vs system optimum", "AHP + TOPSIS"])
    with k1:
        pay = M.payoff_matrix(KPI)
        score = M.uncertainty_criteria(pay)
        picks = M.criteria_picks(score)
        st.markdown("Acts = the four policies, states of nature = the seven scenarios, payoff "
                    "= minus average delay (s/veh).")
        pv = pay.copy()
        pv.columns = [LABEL[c] for c in pv.columns]
        table(pv.round(1).reset_index().rename(columns={"policy": "Policy"}))
        picks_t = picks.rename(columns={"criterion": "Criterion", "picks": "Picks",
                                        "value": "Criterion value"})
        table(picks_t.round(1))
        _pk = picks.groupby("picks").criterion.apply(list)
        st.caption(" · ".join(f"**{p_}**: {', '.join(c_)}" for p_, c_ in _pk.items())
                   + ". The optimist (maximax) looks only at each policy's best case, the mildest "
                   "scenario; the other criteria guard against bad scenarios. Hurwicz uses "
                   "alpha = 0.5.")
    with k2:
        rho = st.slider("Risk aversion rho (per second of delay)", 0.0, 0.5, 0.10, 0.05,
                        key="rho")
        ra_t = M.risk_averse_table(KPI, rho)
        ra_t["scenario"] = ra_t.scenario.map(LABEL)
        ra_t = ra_t.rename(columns={"scenario": "Scenario", "policy": "Policy",
                                    "mean_delay": "Mean delay (s)", "sd": "SD (s)",
                                    "worst": "Worst run (s)",
                                    "certainty_equivalent": "Certainty equivalent (s)"})
        table(ra_t.round(1), height=360)
        st.caption("Utility u(d) = -exp(rho d) over the delay of each of the 8 test runs; the "
                   "certainty equivalent is the sure delay with the same utility. A risk-averse "
                   "planner penalises policies whose runs vary more.")
    with k3:
        poa = M.price_of_anarchy(T["abl"])
        st.markdown("Each AV minimising its own travel time (selfish) is the Nash-like outcome; "
                    "cooperative routing adds the delay a vehicle imposes on others (a proxy "
                    "for the system optimum, not a computed optimum). Both use the same "
                    "80-episode adaptation budget.")
        pt = poa.copy()
        pt["scenario"] = pt.scenario.map(LABEL)
        pt = pt.rename(columns={"scenario": "Scenario", "static_routes": "Static routes (s)",
                                "selfish_routing": "Selfish RL (s)",
                                "cooperative_routing": "Cooperative RL (s)",
                                "price_of_anarchy": "Price of anarchy"})
        table(pt.round(3))
        st.caption("Price of anarchy = selfish delay / cooperative delay: about 1.10 in rush "
                   "hour, about 1.00 when roads are not congested.")
    with k4:
        ah = M.ahp()
        w = ah["weights"]
        cA, cB = st.columns([1, 1.4])
        with cA:
            st.markdown("**AHP weights** (pairwise judgements on the Saaty 1-9 scale)")
            table(pd.DataFrame(dict(Criterion=list(w), Weight=[round(float(v), 3) for v in w.values()])))
            st.caption(f"lambda_max = {ah['lambda_max']:.3f}, consistency index = {ah['ci']:.3f}, "
                       f"consistency ratio = {ah['cr']:.3f} (acceptable below 0.10).")
            table(pd.DataFrame(M.AHP_MATRIX, columns=M.CRITERIA, index=M.CRITERIA)
                  .round(2).reset_index().rename(columns={"index": ""}))
        with cB:
            st.markdown("**TOPSIS closeness to the ideal policy** (1 = ideal)")
            tp = M.topsis_all(KPI, w)
            tp["scenario"] = tp.scenario.map(LABEL)
            table(tp.rename(columns={"scenario": "Scenario", "best": "Best"}).round(3))
        st.caption("TOPSIS uses delay, safety score and CO2 with the AHP weights. It agrees "
                   "with the decision network that Actuated is best in calm conditions and "
                   "the RL packages in rush hour and the event surge.")

    # ---- value of time ---------------------------------------------------------------
    st.markdown("#### What the savings are worth: value of time in AED")
    vc = st.columns(4)
    vot = vc[0].number_input("Value of one person-hour (AED)", 10.0, 250.0, float(V.VOT_AED_DEFAULT),
                             1.0, key="vot-aed",
                             help="Default: USD 14.10 per hour (U.S. DOT, 2015 dollars) x 3.6725. "
                                  "The U.S. range 10.00-17.00 USD is 37-62 AED.")
    occ = vc[1].number_input("Persons per vehicle", 1.0, 3.0, float(V.OCCUPANCY_DEFAULT), 0.1,
                             key="vot-occ", help="1.0 = driver only (conservative).")
    wins = vc[2].number_input("Peak windows per year", 0, 2000, int(V.WINDOWS_PER_YEAR_DEFAULT), 50,
                              key="vot-win", help="Assumption: two measured peak windows on each of "
                                                  "250 working days. Each window is the 25 minutes "
                                                  "the KPIs cover.")
    base_pol = vc[3].selectbox("Compare against", ["Fixed-time", "Actuated"], key="vot-base")
    sv = V.savings_table(KPI, base_pol, vot, occ, wins)
    one = sv[(sv.scenario == scenario) & (sv.policy == policy)]
    mc = st.columns(3)
    if policy == base_pol or one.empty:
        mc[0].info(f"Pick a policy other than {base_pol} in the sidebar to see its savings.")
    else:
        r = one.iloc[0]
        mc[0].metric(f"Vehicle-hours saved ({LABEL[scenario]})", f"{r.saved_veh_h:,.1f} h",
                     f"{r.saved_pct:.0f}% of the delay", delta_color="off")
        mc[1].metric("Value per peak window", f"AED {r.saved_aed:,.0f}",
                     f"± {r.saved_aed_sd:,.0f} (sd over 8 seeds)", delta_color="off")
        mc[2].metric("Value per year (illustrative)", f"AED {r.saved_aed_per_year:,.0f}",
                     f"{wins:,} windows", delta_color="off")
    v1, v2 = st.columns([1, 1.15])
    with v1:
        sub = sv[(sv.scenario == scenario) & (sv.policy != base_pol)].sort_values(
            "saved_aed", ascending=False)
        show(viz.money_bars(sub, "policy", "saved_aed", "saved_aed_sd", height=300,
                            title=f"{LABEL[scenario]}: value of time saved vs {base_pol}",
                            colors=viz.POLICY_COLORS), key="vot-bars")
    with v2:
        pv = sv[sv.policy != base_pol].pivot(index="scenario", columns="policy",
                                             values="saved_aed").reindex(C.SCENARIO_ORDER)
        pv.index = [LABEL[x] for x in pv.index]
        st.markdown(f"**AED saved per peak window vs {base_pol}, every scenario**")
        table(pv.round(0).rename_axis("Scenario").reset_index(), height=300)
    ab_ = T["abl"]
    sx_ = ab_[ab_.study == "Signal x routing"]
    if scenario in sx_.scenario.unique():
        st.markdown(f"**Where the money comes from ({LABEL[scenario]}):** every signal × routing "
                    "pair against fixed-time signals with static routes")
        vs = V.variant_savings(ab_, scenario, "Fixed-time + Static", vot, occ, wins)
        w1, w2 = st.columns([1.3, 1])
        with w1:
            show(viz.money_bars(vs, "variant", "saved_aed", "saved_aed_sd", height=360,
                                colors={v: viz.POLICY_COLORS[v.split(" + ")[0]]
                                        for v in vs.variant}, title="AED saved per peak window"),
                 key="vot-variants")
        with w2:
            table(vs.rename(columns={"variant": "Signals + routing", "delay_s": "Delay (s/veh)",
                                     "saved_veh_h": "Veh-h saved", "saved_aed": "AED / window",
                                     "saved_aed_sd": "sd", "saved_aed_per_year": "AED / year"})
                  .round(1), height=360)
    st.caption("Value of time = hours saved × value of an hour. " + V.SOURCE_TEXT +
               " Hours saved = vehicles × average delay, paired by evaluation seed. One window "
               "is the 25 minutes of an episode that the KPIs cover on this 9-junction grid, so "
               "the yearly figure is an extrapolation under the stated assumption, not a forecast. "
               "Savings against fixed-time signals mix a signal effect and a routing effect; "
               "the second chart separates them. Emissions, safety and bus passengers are not "
               "valued here.")

    # ---- pedestrian trade-off ----------------------------------------------------------
    st.markdown("#### Pedestrian trade-off: what does protecting pedestrians cost?")
    ped_raw = T.get("ped")
    if ped_raw is None:
        st.info("Run `python ped_tradeoff.py` to create data/ped_weight_tradeoff.csv.")
    else:
        pc = st.columns([1, 1, 2])
        wm = pc[0].number_input("Pedestrian time counts × vehicle time", 0.5, 5.0,
                                float(V.WAIT_MULT_DEFAULT), 0.5, key="ped-mult",
                                help="Waiting time is valued at twice in-vehicle time in the "
                                     "U.S. DOT guidance (100% vs 50% of the wage).")
        pol_ped = pc[1].selectbox("Learner", [p for p in ["Q-learning", "SARSA"]
                                              if p in ped_raw.policy.unique()], key="ped-pol")
        ts = V.tradeoff_summary(ped_raw, vot_aed=vot, occupancy=occ, wait_mult=wm)
        p1, p2 = st.columns([1.3, 1])
        with p1:
            show(viz.ped_tradeoff_chart(ts, height=420,
                                        title="Rush hour: vehicle delay vs pedestrian wait, "
                                              "by pedestrian weight in the reward"),
                 key="ped-frontier")
        with p2:
            show(viz.net_benefit_bars(ts, pol_ped, height=420,
                                      title="Same sweep in money (vs actuated signals)"),
                 key="ped-net")
        with st.expander("Table view"):
            tt_ = ts[ts.policy == pol_ped][["ped_weight", "runs", "delay", "delay_sd", "ped_wait",
                                            "ped_wait_sd", "ped_over30", "jaywalks", "conflicts",
                                            "net_aed", "net_aed_sd"]]
            table(tt_.round(2))
        sub_ = ts[ts.policy == pol_ped].sort_values("ped_weight")
        if len(sub_) >= 2:
            lo, hi = sub_.iloc[0], sub_.iloc[-1]
            st.markdown(f"**Reading the sweep ({pol_ped}):** from W_PED = {lo.ped_weight:g} to "
                        f"{hi.ped_weight:g}, pedestrian wait moves {lo.ped_wait:.1f} → "
                        f"{hi.ped_wait:.1f} s and vehicle delay {lo.delay:.1f} → {hi.delay:.1f} s "
                        f"(means over {int(lo.runs)} training runs).")
        st.caption("W_PED is the weight of one pedestrian-second against one vehicle-second in "
                   "the signal agents' reward (0.5 in every other result in this dashboard). "
                   "For each weight the system was retrained from scratch (pre-training on all "
                   "scenarios, then adaptation to rush hour, general vs adapted table chosen on "
                   "validation episodes) and evaluated on the same 8 held-out test episodes. "
                   "Whiskers are one standard deviation across training runs, which is as large "
                   "as many of the differences between neighbouring weights: read the overall "
                   "slope, not single points. Reference markers use the same cooperative routing, "
                   "so only the signal controller differs. The money chart counts vehicle-hours "
                   "saved and pedestrian-hours saved at the values set above. Retraining at 0.5 gives "
                   "about 69 s rush-hour delay on average (64.8, 72.4 and 70.1 s over three training "
                   "runs), not the 64.8 s of the deployed model: the gap is training randomness.")

# ======================================================================================
# 6. Policy comparison
# ======================================================================================
if view == VIEWS[5]:
    st.markdown("#### Headline: signals or routing?")
    _ab = T["abl"]
    _sx = _ab[_ab.study == "Signal x routing"]
    if len(_sx):
        def _v(sc, var):
            return _sx[(_sx.scenario == sc) & (_sx.variant == var)].sort_values("seed").avg_delay_s
        try:
            from scipy import stats as _st
            _p = lambda x, y: float(_st.ttest_rel(x, y).pvalue)
        except Exception:
            _p = lambda x, y: float("nan")
        rq, ra = _v("rush_hour", "Q-learning + Cooperative RL"), _v("rush_hour", "Actuated + Cooperative RL")
        eq, ea = _v("event_surge", "Q-learning + Cooperative RL"), _v("event_surge", "Actuated + Cooperative RL")
        f0, f1 = _v("rush_hour", "Fixed-time + Static").mean(), _v("rush_hour", "Fixed-time + Cooperative RL").mean()
        hc = st.columns(3)
        hc[0].metric("Rush hour: routing alone (fixed-time signals)", f"{f1:.1f} s",
                     f"{pct(f1, f0):+.0f}% vs static routes ({f0:.1f} s)", delta_color="inverse")
        hc[1].metric("Rush hour, same routing: Q-learning vs actuated signals",
                     f"{rq.mean():.1f} vs {ra.mean():.1f} s",
                     f"{pct(rq.mean(), ra.mean()):+.0f}% (paired p = {_p(rq, ra):.2f})",
                     delta_color="off")
        hc[2].metric("Event surge, same routing: Q-learning vs actuated signals",
                     f"{eq.mean():.1f} vs {ea.mean():.1f} s",
                     f"{pct(eq.mean(), ea.mean()):+.0f}% (paired p {'< 0.001' if _p(eq, ea) < 0.001 else '= %.3f' % _p(eq, ea)})",
                     delta_color="inverse")
        st.markdown("**Cooperative routing is the big lever.** Against the actuated rule as first "
                    "written, RL signals win the event surge; in the daily peak, actuated signals "
                    "with the same routing are statistically as good. Against a *tuned* actuated "
                    "controller (below), RL signals tie: they learn expert-level timing without "
                    "manual tuning. This matches the decision network, which recommends actuated "
                    "signals + cooperative routing in 6 of 7 scenarios (22 of 27 weather × "
                    "density × pedestrian conditions).")
        st.info("Fair-comparison check (Research Extensions): with its four parameters tuned on "
                "validation runs, the actuated controller ties Q-learning in the event surge and "
                "beats it in rush hour. RL signals reach tuned-expert performance without manual "
                "tuning; they do not beat a well-tuned actuated plan.")
        st.caption("The policies elsewhere in this dashboard are packages: *Actuated* = actuated "
                   "signals + static routes; *Q-learning* / *SARSA* = RL signals + cooperative "
                   "routing. Comparing signal controllers fairly means giving both the same routing.")
    st.markdown("#### Which part of the system delivers the gain? Signal timing × routing")
    ab = T["abl"]
    sxr = ab[ab.study == "Signal x routing"]
    if len(sxr):
        scs = [x for x in C.SCENARIO_ORDER if x in sxr.scenario.unique()]
        pick = st.selectbox("Scenario", scs, index=scs.index(scenario) if scenario in scs else 0,
                            format_func=lambda x: LABEL[x], key="sxr-sc")
        cx1, cx2 = st.columns([1.3, 1])
        with cx1:
            show(viz.combo_bars(sxr[sxr.scenario == pick], height=340,
                                title=f"Every signal × routing pair, {LABEL[pick]} (8 seeds)"),
                 key="sxr")
        with cx2:
            tb = sxr[sxr.scenario == pick].groupby("variant")[
                ["avg_delay_s", "conflicts_per_1000", "co2_kg", "jaywalks"]].mean() \
                .sort_values("avg_delay_s").round(2).reset_index()
            table(tb, height=340)
        st.caption("Cooperative RL routing = the Q-learning AV fleet (SARSA fleet with SARSA "
                   "signals). These are the decision network's eight options, simulated "
                   "directly on the evaluation seeds.")
    if T["sel"] is not None:
        st.markdown("#### Model selection: general vs scenario-adapted tables (validation runs)")
        sl = T["sel"].copy()
        sl["scenario"] = sl.scenario.map(LABEL)
        sl = sl.rename(columns={"policy": "Learner", "scenario": "Scenario",
                                "validation_general_s": "Validation, general (s)",
                                "validation_adapted_s": "Validation, adapted (s)",
                                "chosen": "Deployed", "test_general_s": "Test, general (s)",
                                "test_adapted_s": "Test, adapted (s)"})
        table(sl.round(1))
        st.caption("The deployed table is chosen on separate validation episodes (seeds "
                   "800,000+); test episodes (900,000+) are only used to report. In rush hour "
                   "the extra scenario-specific training made both learners worse.")
    st.markdown("#### All policies, all scenarios")
    metric = st.selectbox("Measure", ["avg_delay_s", "throughput_vph", "co2_kg",
                                      "conflicts_per_1000", "safety_score", "ped_avg_wait_s",
                                      "jaywalks", "bus_on_time_pct", "bus_avg_delay_s",
                                      "nox_g", "idle_hours", "avg_stops", "ev_avg_delay_s",
                                      "ped_over30_pct", "switches"],
                          format_func=lambda m: MEASURE.get(m, m),
                          key="pc-m")
    show(viz.policy_bars(KPI, metric, MEASURE.get(metric, metric), height=380,
                         title=f"{MEASURE.get(metric, metric)}: mean ± sd over 8 seeds"),
         key="pc-bars")
    st.markdown("#### Q-learning vs SARSA (paired by evaluation seed)")
    rows = []
    try:
        from scipy import stats
    except Exception:
        stats = None
    for s_ in C.SCENARIO_ORDER:
        a = KPI[(KPI.scenario == s_) & (KPI.policy == "Q-learning")].sort_values("seed")
        b = KPI[(KPI.scenario == s_) & (KPI.policy == "SARSA")].sort_values("seed")
        d = b.avg_delay_s.values - a.avg_delay_s.values
        p = float(stats.ttest_rel(b.avg_delay_s, a.avg_delay_s).pvalue) if stats else np.nan
        rows.append(dict(scenario=LABEL[s_], q_learning_delay=a.avg_delay_s.mean(),
                         sarsa_delay=b.avg_delay_s.mean(), sarsa_minus_q=d.mean(),
                         paired_t_p=p, q_learning_conflicts=a.conflicts_per_1000.mean(),
                         sarsa_conflicts=b.conflicts_per_1000.mean(),
                         q_learning_jaywalks=a.jaywalks.mean(), sarsa_jaywalks=b.jaywalks.mean()))
    table(pd.DataFrame(rows).round(3))
    st.caption("Negative sarsa_minus_q = SARSA has less delay. Common random numbers make "
               "the comparison paired: each seed has identical arrivals for both policies. "
               "The p-value covers evaluation noise only: the Q-learning / SARSA gap mostly "
               "comes from training randomness (re-adapting with other seeds flips the order "
               "in rush hour, table below).")
    if T["robust"] is not None:
        st.markdown("#### Training stability (re-adapting with different random seeds)")
        r = T["robust"].groupby(["scenario", "policy"]).avg_delay_s.agg(
            ["mean", "std", "min", "max"]).reset_index()
        r["scenario"] = r.scenario.map(LABEL)
        table(r.round(1))
    st.markdown("#### Component studies (Q-learning signals)")
    a1, a2 = st.columns(2)
    for col, study, metrics in [(a1, "AV routing", ["avg_delay_s", "av_avg_delay_s"]),
                                (a2, "Bus priority", ["bus_avg_delay_s", "bus_on_time_pct",
                                                      "avg_delay_s"])]:
        with col:
            st.markdown(f"**{study}**")
            sub = ab[ab.study == study].groupby(["scenario", "variant"])[metrics].mean() \
                .round(1).reset_index()
            sub["scenario"] = sub.scenario.map(LABEL)
            table(sub)
    st.markdown("#### Learning curves")
    l1, l2 = st.columns(2)
    with l1:
        show(viz.learning_curves(T["curves"], "signal_return", "pre-train", height=320,
                                 label="Episode return (signal agents)",
                                 title="Return during pre-training"), key="lc3")
    with l2:
        show(viz.learning_curves(T["curves"], "conflicts_per_1000", "pre-train", height=320,
                                 label="Expected conflicts per 1,000 veh",
                                 title="Safety during pre-training (exploring)"), key="lc4")


# ======================================================================================
# 7. Simulation Lab (live what-if with correlated conditions)
# ======================================================================================
LAB_KEYS = dict(demand="lab-d", ped="lab-p", pax="lab-pax", weather="lab-w", peak="lab-peak",
                av_share="lab-av", incident="lab-inc", inc_loss="lab-loss", evs="lab-ev",
                bus_lane="lab-lane", headway_min="lab-hw", event="lab-evt",
                glosa="lab-glosa", av_lane="lab-avlane", ped_guard="lab-guard",
                sense_bias="lab-sbias", sense_noise="lab-snoise")
for _f, _k in LAB_KEYS.items():
    st.session_state.setdefault(_k, LB.DEFAULTS[_f])
st.session_state.setdefault("lab-link", True)
st.session_state.setdefault("lab-preset", "Custom")
st.session_state.setdefault("lab-moved", [])


def _ss(f):
    return st.session_state[LAB_KEYS[f]]


def _set(f, v, moved):
    if st.session_state[LAB_KEYS[f]] != v:
        st.session_state[LAB_KEYS[f]] = v
        moved.append(f)


def _lab_couple(driver):
    """Move the followers of `driver` (linked mode only)."""
    st.session_state["lab-preset"] = "Custom"
    if not st.session_state.get("lab-link"):
        st.session_state["lab-moved"] = []
        return
    moved = []
    if driver == "peak":
        _set("demand", LB.PEAK["demand"] if _ss("peak") else 1.0, moved)
        driver = "demand"
    if driver in ("demand", "all"):
        _set("ped", LB.linked_ped(_ss("demand")), moved)
        _set("pax", LB.linked_pax(_ss("demand"), _ss("event")), moved)
    if driver in ("event", "all"):
        _set("pax", LB.linked_pax(_ss("demand"), _ss("event")), moved)
    if driver in ("weather", "incident", "all"):
        _set("evs", LB.linked_evs(_ss("weather"), _ss("incident")), moved)
    if driver in ("bus_lane", "all"):
        _set("headway_min", LB.BUS_LANE_HEADWAY_MIN if _ss("bus_lane") else 6, moved)
    st.session_state["lab-moved"] = moved
    st.session_state["lab-preset"] = "Custom"


def _lab_custom():
    st.session_state["lab-preset"] = "Custom"
    st.session_state["lab-moved"] = []


def _lab_preset():
    key = st.session_state["lab-preset"]
    if key == "Custom":
        return
    sc = C.SCENARIOS[key]
    vals = dict(demand=sc["demand_mult"], ped=sc["ped_mult"], pax=sc["pax_mult"],
                weather=sc["weather"], peak=sc["profile"] == "peak", av_share=C.AV_SHARE,
                incident=bool(sc.get("incident")), inc_loss=75, evs=int(sc["ev_per_episode"]),
                bus_lane=bool(sc.get("bus_lane")), headway_min=int(sc["bus_headway_s"] // 60),
                event=bool(sc.get("event")))
    for f, v in vals.items():
        st.session_state[LAB_KEYS[f]] = v
    st.session_state["lab-moved"] = []


def _lab_link():
    if st.session_state.get("lab-link"):
        _lab_couple("all")


@st.cache_data(show_spinner=False, max_entries=64)
def _lab_run(cond_items, n_seeds):
    return LB.run(load_models(), dict(cond_items), n_seeds=n_seeds)


@st.cache_data(show_spinner=False, max_entries=16)
def _lab_record(cond_items, pols):
    return LB.record(load_models(), dict(cond_items), list(pols))


@st.cache_data(show_spinner=False, max_entries=16)
def _lab_sweep(cond_items, var, linked):
    return LB.sweep(load_models(), dict(cond_items), var, linked)


if view == VIEWS[6]:
    st.markdown("#### Simulation Lab: set the city's conditions, every controller re-runs live")
    st.caption("Each change simulates 30 minutes for all four policies on identical arrivals "
               "(common random numbers). RL policies use the general tables, pre-trained on "
               "every scenario, because custom conditions match no single scenario.")
    top = st.columns([1.3, 1.3, 1, 1])
    top[0].selectbox("Start from a scenario", ["Custom"] + C.SCENARIO_ORDER, key="lab-preset",
                     format_func=lambda k: "Custom conditions" if k == "Custom" else LABEL[k],
                     on_change=_lab_preset,
                     help="Loads that scenario's exact settings into the controls below.")
    with top[1]:
        st.toggle("Link correlated conditions", key="lab-link", on_change=_lab_link,
                  help="Conditions move together as they do in the city: demand pulls "
                       "pedestrians and bus passengers with it, bad weather and crashes bring "
                       "more emergency calls, a bus lane comes with a 3-minute service. See the "
                       "correlation map at the bottom.")
    n_seeds = top[2].select_slider("Seeds per policy", [2, 3, 4, 6, 8], value=3, key="lab-seeds",
                                   help="More seeds = less noise, slower (≈0.06 s per episode).")
    with top[3]:
        st.write("")
        if st.button("↺ Reset to normal flow", width="stretch"):
            for _f, _k in LAB_KEYS.items():
                st.session_state[_k] = LB.DEFAULTS[_f]
            st.session_state["lab-moved"] = []
            st.rerun()

    g1, g2, g3 = st.columns(3)
    with g1.container(border=True):
        st.markdown("**🚗 Demand & people**")
        st.slider("Vehicle demand (× normal)", 0.5, 1.8, step=0.05, key="lab-d",
                  on_change=_lab_couple, args=("demand",))
        st.toggle("Rush-hour ramp (peaks mid-episode)", key="lab-peak", on_change=_lab_couple,
                  args=("peak",))
        st.slider("Pedestrians (× normal)", 0.25, 4.0, step=0.05, key="lab-p", on_change=_lab_custom)
        st.slider("Bus passengers (× normal)", 0.5, 3.0, step=0.1, key="lab-pax", on_change=_lab_custom)
    with g2.container(border=True):
        st.markdown("**🌪 Environment & disruptions**")
        st.selectbox("Weather", C.WEATHER_NAMES, key="lab-w", on_change=_lab_couple,
                     args=("weather",))
        st.toggle("Crash blocks Street 2 eastbound at J5 (min 5-25)", key="lab-inc",
                  on_change=_lab_couple, args=("incident",))
        st.slider("Capacity lost to the crash (%)", 25, 95, step=5, key="lab-loss",
                  on_change=_lab_custom, disabled=not st.session_state["lab-inc"])
        st.toggle("Arena event lets out (minute 8)", key="lab-evt", on_change=_lab_couple,
                  args=("event",))
        st.slider("Emergency calls per 30 min", 0, 10, step=1, key="lab-ev", on_change=_lab_custom)
    with g3.container(border=True):
        st.markdown("**🚌 Fleet & transit**")
        st.slider("Autonomous-vehicle share", 0.0, 0.8, step=0.05, key="lab-av", on_change=_lab_custom,
                  help="Share of vehicles that are connected AVs: they route cooperatively "
                       "under the RL packages and platoon at 0.85 of a car's headway.")
        st.toggle("Dedicated bus lane on Street 2", key="lab-lane", on_change=_lab_couple,
                  args=("bus_lane",))
        st.slider("Bus headway (minutes)", 2, 10, step=1, key="lab-hw", on_change=_lab_custom)
    with st.container(border=True):
        st.markdown("**🛡 Smart-control add-ons** (Research Extensions; apply to every policy)")
        a1, a2, a3 = st.columns(3)
        a1.toggle("AV speed advisory (GLOSA)", key="lab-glosa",
                  help="Connected AVs ease off when the light ahead is red and the queue is "
                       "short, arriving later instead of braking to a stop (max 15 s per link).")
        a2.toggle("AV lane selection", key="lab-avlane",
                  help="AVs pre-select their lane, so they pass a vehicle waiting to turn or "
                       "blocked by a full exit link.")
        a3.select_slider("Pedestrian max-wait guard", [0, 30, 26, 22], key="lab-guard",
                         format_func=lambda v: "off" if v == 0 else f"{v} s",
                         help="Safety supervisor: a signal must serve pedestrians who have waited "
                              "this long, before they give up and cross on red.")
    with st.container(border=True):
        st.markdown("**📡 Sensing uncertainty** (what the signal controllers *see*; the traffic "
                    "itself is unchanged)")
        s1, s2 = st.columns([2, 1])
        s1.select_slider("Queue-count bias", [0.5, 0.75, 1.0, 1.25, 1.5, 2.0], key="lab-sbias",
                         format_func=lambda v: {0.5: "0.5× (half missed)", 0.75: "0.75×",
                                                1.0: "1.0× (exact)", 1.25: "1.25×",
                                                1.5: "1.5× (1.5 phones/person)",
                                                2.0: "2.0× (2 phones/person)"}[v],
                         help="Systematic (likelihood) bias of the queue counts. Counting "
                              "phones and assuming one per person over-counts when people carry "
                              "two; low detector or probe coverage under-counts. Fixed-time "
                              "signals use no detectors and are unaffected.")
        s2.toggle("Random counting noise", key="lab-snoise",
                  help="Each count is drawn from a Poisson distribution around the (biased) "
                       "true value, as with sampled probe data.")

    cond = {f: st.session_state[k] for f, k in LAB_KEYS.items()}
    cond["av_share"] = float(cond["av_share"])
    items = tuple(sorted(cond.items()))
    eff = LB.effective(cond)
    names = dict(demand="demand", ped="pedestrians", pax="bus passengers", evs="emergency calls",
                 headway_min="bus headway")
    chips = [f'<span class="chip">Effective demand <b>{eff["demand"]:.2f}×</b>'
             + (f' → peak <b>{eff["demand_peak"]:.2f}×</b>' if cond["peak"] else "") + '</span>',
             f'<span class="chip">Effective pedestrians <b>{eff["ped"]:.2f}×</b></span>',
             f'<span class="chip">Road capacity (weather) <b>{100 * eff["capacity"]:.0f}%</b></span>',
             f'<span class="chip">Crash risk <b>×{eff["risk"]:.2f}</b></span>']
    if st.session_state.get("lab-link") and st.session_state.get("lab-moved"):
        chips.append('<span class="chip link">🔗 linked: moved '
                     + ", ".join(names.get(f, f) for f in st.session_state["lab-moved"]) + '</span>')
    st.markdown("".join(chips), unsafe_allow_html=True)

    with st.spinner("Simulating 4 policies × %d seeds ..." % n_seeds):
        wi = _lab_run(items, n_seeds)
    g = wi.groupby("policy").mean(numeric_only=True).reindex(POL)
    sc_ = LB.scores(g, WEIGHTS)
    cols = st.columns(4)
    for c, pol_ in zip(cols, POL):
        d_ = g.loc[pol_]
        c.metric(f"{pol_} · score {sc_[pol_]:.0f}/100", f"{d_.avg_delay_s:.1f} s",
                 "baseline" if pol_ == "Fixed-time" else
                 f"{pct(d_.avg_delay_s, g.loc['Fixed-time'].avg_delay_s):+.0f}% delay vs fixed-time",
                 delta_color="off" if pol_ == "Fixed-time" else "inverse",
                 delta_arrow="off" if pol_ == "Fixed-time" else "auto")

    win = sc_.idxmax()
    ev_lab = LB.dn_evidence(cond)
    dn_best, _ = DN.best(ev_lab, WEIGHTS)
    agree = dn_best["Signal timing"] == win
    v1, v2 = st.columns([1, 1])
    v1.markdown(
        f'<div class="verdict"><div class="small">SIMULATION WINNER UNDER YOUR WEIGHTS</div>'
        f'<div class="big">{win} package · score {sc_[win]:.0f}/100</div>'
        f'<div class="small">Weighted sum of delay, safety score and CO₂ (min-max across the four '
        f'packages; weights from the sidebar).</div></div>', unsafe_allow_html=True)
    v2.markdown(
        f'<div class="verdict"><div class="small">DECISION NETWORK RECOMMENDS '
        f'({", ".join(f"{k} = {v}" for k, v in ev_lab.items())})</div>'
        f'<div class="big">{dn_best["Signal timing"]} signals + {dn_best["Routing"]} routing · '
        f'EU {dn_best["EU"]:.1f}</div><div class="small">'
        + ("Same signal controller as the simulation winner." if agree else
           "A different signal controller from the simulation winner: the network averages over "
           "uncertain outcomes learned from all scenarios, the simulation scores these exact "
           "conditions, and the packages also differ in routing.")
        + '</div></div>', unsafe_allow_html=True)

    st.markdown("#### Every criterion at a glance")
    show(viz.lab_bars(g, [("avg_delay_s", "Delay (s/veh)", True),
                          ("safety_score", "Safety score", False),
                          ("co2_kg", "CO₂ (kg)", True),
                          ("nox_g", "NOx (g)", True),
                          ("ped_avg_wait_s", "Pedestrian wait (s)", True),
                          ("bus_on_time_pct", "Bus on time (%)", False),
                          ("throughput_vph", "Veh/h served", False)], height=320),
         key="lab-bars")
    _evt = wi.groupby("policy")[["ev_trips", "ev_avg_delay_s"]].mean().reindex(POL)
    st.markdown("".join(
        f'<span class="chip">🚑 {p_}: <b>{r_.ev_trips:.1f}</b> emergency trips, '
        f'<b>{r_.ev_avg_delay_s:.1f} s</b> delay each</span>' for p_, r_ in _evt.iterrows()),
        unsafe_allow_html=True)
    st.caption(f"Means over {n_seeds} seeds; the outlined bar is the best package on each "
               "criterion. Packages: Actuated = actuated signals + static routes; RL = RL signals "
               "+ cooperative AV routing. With few seeds, gaps of a few percent are noise.")

    st.markdown("#### Watch it: replay of these exact conditions")
    rc = st.columns([1, 1, 2])
    lp = rc[0].selectbox("Left", POL, index=0, key="lab-left")
    rp = rc[1].selectbox("Right", POL, index=2, key="lab-right")
    rc[2].caption("Seed 1 of the run above, first 20 minutes, one clock for both views. "
                  "Press play or space; the badge lights up while an emergency vehicle is "
                  "pre-empting signals.")
    with st.spinner("Recording replays ..."):
        pls = _lab_record(items, (lp, rp))
    embed(sim_view.render_html(pls, [lp, rp], start=0, autoplay=True), height=760)

    st.markdown("#### Stress test: where does each controller break?")
    sv = st.columns([1.2, 2.8])
    var = sv[0].selectbox("Sweep", list(LB.SWEEPS), format_func=lambda k: LB.SWEEPS[k][0],
                          key="lab-sweep-var")
    run_sweep = sv[0].toggle("Run the sweep", key="lab-sweep-on",
                             help="About 3-5 seconds: every policy × every level × 2 seeds.")
    sv[1].caption("Re-runs all four policies along one condition with everything else held at the "
                  "settings above (2 seeds each; bands = range over seeds). In linked mode a "
                  "demand sweep also moves pedestrians and bus passengers, as the sliders do.")
    if run_sweep:
        with st.spinner("Sweeping ..."):
            sw = _lab_sweep(items, var, bool(st.session_state.get("lab-link")))
        cur = {"demand": cond["demand"], "ped": cond["ped"], "av_share": cond["av_share"],
               "inc_loss": cond["inc_loss"] if cond["incident"] else 0}[var]
        s1, s2 = st.columns(2)
        with s1:
            show(viz.sweep_lines(sw, LB.SWEEPS[var][0], "avg_delay_s", "Average delay (s/veh)",
                                 current=cur, title="Delay"), key="sw1")
        with s2:
            show(viz.sweep_lines(sw, LB.SWEEPS[var][0], "conflicts_per_1000",
                                 "Expected conflicts per 1,000 veh", current=cur,
                                 title="Safety (expected conflicts)", fmt=".2f"), key="sw2")
        s3, s4 = st.columns(2)
        with s3:
            show(viz.sweep_lines(sw, LB.SWEEPS[var][0], "co2_kg", "CO₂ (kg)", current=cur,
                                 title="Emissions", fmt=",.0f"), key="sw3")
        with s4:
            show(viz.sweep_lines(sw, LB.SWEEPS[var][0], "ped_avg_wait_s", "Pedestrian wait (s)",
                                 current=cur, title="Pedestrian wait"), key="sw4")

    st.markdown("#### Correlation map: which conditions move together")
    cm1, cm2 = st.columns([1.25, 1])
    with cm1:
        show(viz.correlation_graph(LB.LINKS, active=bool(st.session_state.get("lab-link")),
                                   height=380), key="lab-corr")
    with cm2:
        st.markdown("**Linked by the controls** (" +
                    ("on" if st.session_state.get("lab-link") else "off: sliders move freely")
                    + ")")
        _rows = lambda rs, h: '<div class="rules"><div><span>Driver</span><span>' + h + \
            '</span><span>Rule</span></div>' + "".join(
                f"<div><b>{a_}</b><span>{b_}</span><span>{c_}</span></div>" for a_, b_, c_ in rs) \
            + "</div>"
        st.markdown(_rows(LB.LINKS, "Moves"), unsafe_allow_html=True)
        st.markdown("**Always applied inside the simulator**")
        st.markdown(_rows(LB.INTERNAL, "Effect on"), unsafe_allow_html=True)
    st.caption("Coefficients come from the evaluated scenarios in config.py, so setting a driver "
               "to a scenario's value reproduces that scenario (e.g. the rush-hour ramp gives "
               f"demand 1.30, pedestrians 1.30, bus passengers {LB.linked_pax(1.3):.1f}). The "
               "linked moves are planning assumptions, not measured elasticities; switch linking "
               "off to explore each condition on its own.")

    with st.expander("Full KPI table for these conditions"):
        out = g[["avg_delay_s", "throughput_vph", "avg_stops", "idle_hours", "co2_kg", "nox_g",
                 "conflicts_per_1000", "safety_score", "jaywalks", "ped_avg_wait_s",
                 "bus_on_time_pct", "bus_avg_delay_s", "ev_avg_delay_s", "av_avg_delay_s",
                 "human_avg_delay_s"]]
        table(out.reset_index().rename(columns={"policy": "Policy"}))


# ======================================================================================
# 8. Research extensions (extensions.py): the five review gaps, closed with experiments
# ======================================================================================
def _gap(num, title, verdict, good=True):
    col = "#21c98b" if good else "#f5b301"
    st.markdown(f'<div class="card" style="border-left:4px solid {col}; margin-top:18px">'
                f'<span class="tag" style="color:{col}">GAP {num}</span>'
                f'<h4>{title}</h4><p>{verdict}</p></div>', unsafe_allow_html=True)


if view == VIEWS[7]:
    st.markdown("#### Research extensions: six open questions, answered by experiment")
    st.caption("Every study uses the deployed Q-tables (no retraining), the 8 held-out TEST "
               "episodes for reporting and, where something is tuned, separate VALIDATION "
               "episodes for choosing. New mechanisms are switched on per run, so all other "
               "results in this dashboard are unchanged. Regenerate with `python extensions.py` and `python sensing.py`.")
    if T.get("x_av") is None:
        st.warning("Run `python extensions.py` to create the data/ext_*.csv files.")
        st.stop()
    SCL = {k: LABEL[k] for k in C.SCENARIO_ORDER}

    # ---- 1. AV speed control and lane selection ------------------------------------
    av = T["x_av"]
    gv = av.groupby("variant")[["avg_delay_s", "avg_stops", "co2_kg", "conflicts_per_1000",
                                "av_avg_delay_s", "human_avg_delay_s"]].mean()
    b0 = gv.loc["Baseline"]
    _p = lambda v, c: 100 * (gv.loc[v, c] - b0[c]) / b0[c]
    _gap(1, "Autonomous vehicles: speed control and lane selection",
         f"Two V2I behaviours added to the AV agents. <b>Lane selection</b> (AVs pass a vehicle "
         f"waiting to turn or blocked by a full exit): AV delay {_p('+ lane selection', 'av_avg_delay_s'):+.1f}%, "
         f"network delay {_p('+ lane selection', 'avg_delay_s'):+.1f}%. <b>Speed advisory</b> "
         f"(ease off towards a red light instead of stopping): stops "
         f"{_p('+ speed advisory', 'avg_stops'):+.1f}%, conflicts "
         f"{_p('+ speed advisory', 'conflicts_per_1000'):+.1f}%, CO₂ "
         f"{_p('+ speed advisory', 'co2_kg'):+.1f}%, paid for with "
         f"{_p('+ speed advisory', 'av_avg_delay_s'):+.1f}% AV delay. Means over all scenarios; "
         "effects are small at a 25 % AV share and grow with it (try the AV-share slider in the "
         "Simulation Lab with both add-ons on).")
    c1, c2 = st.columns(2)
    order_v = list(AV_ORDER := ["Baseline", "+ speed advisory", "+ lane selection", "+ both"])
    with c1:
        show(viz.grouped_bars(av, "scenario", "variant", "avg_delay_s", "s/veh",
                              order_x=C.SCENARIO_ORDER, order_g=order_v, x_labels=SCL,
                              title="Average delay (Q-learning package)", height=340),
             key="x-av1")
    with c2:
        tb = gv.loc[order_v].rename(columns={"avg_delay_s": "Delay (s)", "avg_stops": "Stops / veh",
                                             "co2_kg": "CO₂ (kg)",
                                             "conflicts_per_1000": "Conflicts / 1,000",
                                             "av_avg_delay_s": "AV delay (s)",
                                             "human_avg_delay_s": "Human-driven delay (s)"})
        st.markdown("**All scenarios, mean of 8 test episodes each**")
        table(tb.round(3).reset_index().rename(columns={"variant": "AV behaviour"}))

    # ---- 2. Bus passenger load balancing ---------------------------------------------
    bus = T["x_bus"]
    gb = bus.groupby(["scenario", "variant"])[["bus_load_cv", "bus_load_p95_pct",
                                               "pax_left_behind", "bus_on_time_pct"]].mean()
    rh = gb.loc["rush_hour"]
    _gap(2, "Public transport: passenger load balancing",
         "Bus loads are now measured (load at every stop departure, its spread, the 95th "
         "percentile and passengers left behind by full buses) and the learned bus agents are "
         "compared with an even-headway holding rule. Finding: in rush hour the busiest buses "
         f"run full (95th percentile {rh.loc['RL bus agents (deployed)', 'bus_load_p95_pct']:.0f} % "
         f"of capacity) and about {rh.loc['RL bus agents (deployed)', 'pax_left_behind']:.0f} "
         "passengers per window are left behind <i>whatever</i> the holding strategy: the problem "
         "is capacity, not balance. A bus every 3 minutes cuts that to "
         f"{rh.loc['RL bus agents + bus every 3 min', 'pax_left_behind']:.1f}. The RL agents keep "
         f"the best on-time record ({rh.loc['RL bus agents (deployed)', 'bus_on_time_pct']:.0f} % vs "
         f"{rh.loc['No priority, no holding', 'bus_on_time_pct']:.0f} % without priority or holding).")
    bv = ["No priority, no holding", "Always request priority", "Even-headway holding rule",
          "RL bus agents (deployed)", "RL bus agents + bus every 3 min"]
    bsc = ["normal", "rush_hour", "bus_corridor"]
    c1, c2, c3 = st.columns(3)
    with c1:
        show(viz.grouped_bars(bus, "scenario", "variant", "pax_left_behind", "passengers",
                              order_x=bsc, order_g=bv, x_labels=SCL, height=360,
                              title="Passengers left behind by full buses"), key="x-b1")
    with c2:
        show(viz.grouped_bars(bus, "scenario", "variant", "bus_load_p95_pct", "% of capacity",
                              order_x=bsc, order_g=bv, x_labels=SCL, height=360,
                              title="Busiest buses (95th percentile load)"), key="x-b2")
    with c3:
        show(viz.grouped_bars(bus, "scenario", "variant", "bus_on_time_pct", "% on time",
                              order_x=bsc, order_g=bv, x_labels=SCL, height=360,
                              title="Bus on-time performance"), key="x-b3")
    st.caption("The bus-priority corridor already runs a bus every 3 minutes, so its last two "
               "bars are identical. Load spread (coefficient of variation) stays at 0.12-0.22 "
               "for every strategy; see the table view.")
    with st.expander("Table view: bus loads"):
        table(gb.round(2).reset_index())

    # ---- 3. Proxy safety model ------------------------------------------------------
    rk = T["x_rank"]
    nonbase = rk[rk.variant != "Base model"]
    _gap(3, "Safety is a proxy: does the conclusion depend on its parameters?",
         "Each of the three conflict mechanisms (red-light running, rear-end exposure from "
         "stops, jaywalking) was halved and doubled, and all four policies re-run in every "
         f"scenario. The safest policy is unchanged in {100 * nonbase.same_safest.mean():.0f} % "
         f"of the {len(nonbase)} scenario × variant cases, and where it changes it only swaps "
         "between the two non-learning controllers. The RL packages are never the safest under "
         "any variant, so the safety conclusion does not hinge on the uncertain parameters. "
         "Conflicts remain a surrogate for crashes; the ranking, not the absolute number, is "
         "what the project relies on.")
    sv = T["x_safe"]
    c1, c2 = st.columns([1.3, 1])
    with c1:
        pick = st.selectbox("Scenario", C.SCENARIO_ORDER, index=1, format_func=lambda k: LABEL[k],
                            key="x-safe-sc")
        show(viz.grouped_bars(sv[sv.scenario == pick], "variant", "policy",
                              "conflicts_per_1000", "conflicts / 1,000 veh",
                              order_x=list(dict.fromkeys(sv.variant)), order_g=POL,
                              colors=viz.POLICY_COLORS, height=360,
                              title=f"Expected conflicts under each safety-model variant, {LABEL[pick]}"),
             key="x-s1")
    with c2:
        st.markdown("**Cases where the safest policy changes**")
        ch = rk[~rk.same_safest].copy()
        ch["scenario"] = ch.scenario.map(LABEL)
        table(ch[["scenario", "variant", "ranking"]] if len(ch) else
              pd.DataFrame({"result": ["none: same safest policy everywhere"]}), height=220)
        st.caption("Ranking = safest first. Value of time: savings scale linearly with the AED "
                   "per hour set in the KPI tab, so no ranking depends on the U.S. figure.")

    # ---- 4. Fair comparison -------------------------------------------------------
    ta = T["x_ta"]
    co = ta[ta.routing == "Cooperative RL"].groupby(["scenario", "controller"]).avg_delay_s.mean()
    rob = T["robust"]
    rr = rob[(rob.policy == "Q-learning")].groupby("scenario").avg_delay_s.agg(["mean", "std"]) \
        if rob is not None else None
    _gap(4, "A fair baseline: tune the actuated controller, average RL over training runs",
         "The actuated controller's four parameters (max green, queue ratio, queue offset, "
         "pedestrian service time) were tuned per scenario on validation episodes from 81 "
         "combinations, then tested with the same cooperative routing as RL. Result: "
         f"<b>event surge</b> {co[('event_surge', 'Q-learning (deployed)')]:.1f} s (Q-learning) vs "
         f"{co[('event_surge', 'Actuated (tuned)')]:.1f} s (tuned actuated), a tie; <b>rush hour</b> "
         f"{co[('rush_hour', 'Q-learning (deployed)')]:.1f} vs {co[('rush_hour', 'Actuated (tuned)')]:.1f} s, "
         "tuned actuated is better. Averaged over three re-trainings, Q-learning's rush-hour delay is "
         + (f"{rr.loc['rush_hour', 'mean']:.1f} ± {rr.loc['rush_hour', 'std']:.1f} s" if rr is not None else "n/a")
         + ". Honest reading: RL signals learn tuned-expert performance on their own, without "
         "manual tuning, but they do not beat a well-tuned actuated plan; cooperative AV routing "
         "is the gain that survives a fair comparison.", good=False)
    c1, c2 = st.columns([1.4, 1])
    with c1:
        show(viz.grouped_bars(ta[ta.routing == "Cooperative RL"], "scenario", "controller",
                              "avg_delay_s", "s/veh", order_x=C.SCENARIO_ORDER,
                              order_g=["Actuated (untuned)", "Actuated (tuned)",
                                       "Q-learning (deployed)"],
                              colors={"Actuated (untuned)": "#5f8f7d", "Actuated (tuned)": "#21c98b",
                                      "Q-learning (deployed)": "#3d8bff"}, x_labels=SCL,
                              title="Same cooperative routing: tuned vs untuned actuated vs RL",
                              height=360), key="x-t1")
    with c2:
        st.markdown("**Tuned parameters (chosen on validation runs)**")
        tt = T["x_tune"].copy()
        tt["scenario"] = tt.scenario.map(LABEL)
        table(tt.rename(columns={"scenario": "Scenario", "max_green": "Max green (s)",
                                 "ratio": "Queue ratio", "offset": "Queue offset",
                                 "ped_after": "Ped service after (s)",
                                 "validation_delay_s": "Validation delay (s)"}).round(1), height=290)
        if rr is not None:
            st.caption("Q-learning re-trained with three other seeds: " + ", ".join(
                f"{LABEL[k]} {v['mean']:.1f} ± {v['std']:.1f} s" for k, v in rr.iterrows()))

    # ---- 5. RL safety at high demand ---------------------------------------------
    gd = T["x_guard"]
    gg = gd.groupby(["condition", "variant"])[["avg_delay_s", "conflicts_per_1000",
                                               "ped_avg_wait_s", "jaywalks"]].mean()
    stc = "Stress: peak at 1.6 × demand"
    _gap(5, "RL safety at high demand: a pedestrian max-wait guard",
         "Diagnosis: at extreme demand RL's extra conflicts come from jaywalking, because it "
         f"leaves pedestrians waiting {gg.loc[(stc, 'Q-learning'), 'ped_avg_wait_s']:.0f} s on average "
         "(non-compliance starts beyond 30 s). Fix: a safety supervisor on top of the RL signals "
         "forces a crossing to be served once its first pedestrian has waited a set time. Under "
         f"stress it cuts conflicts from {gg.loc[(stc, 'Q-learning'), 'conflicts_per_1000']:.1f} to "
         f"{gg.loc[(stc, 'Q-learning + guard 30 s'), 'conflicts_per_1000']:.1f} per 1,000 vehicles "
         f"(actuated: {gg.loc[(stc, 'Actuated package (reference)'), 'conflicts_per_1000']:.1f}) while staying "
         f"faster than actuated ({gg.loc[(stc, 'Q-learning + guard 30 s'), 'avg_delay_s']:.0f} vs "
         f"{gg.loc[(stc, 'Actuated package (reference)'), 'avg_delay_s']:.0f} s). The price is delay in "
         "rush hour, and in the event surge the guard is counter-productive (thousands of "
         "pedestrians force constant switching), so it should be switched on by demand level: "
         "exactly the kind of choice the decision network makes.")
    gvars = ["Q-learning", "Q-learning + guard 30 s", "Q-learning + guard 26 s",
             "Q-learning + guard 22 s", "Actuated package (reference)"]
    gcol = {"Q-learning": "#3d8bff", "Q-learning + guard 30 s": "#22d3ee",
            "Q-learning + guard 26 s": "#7dd3fc", "Q-learning + guard 22 s": "#a78bfa",
            "Actuated package (reference)": "#21c98b"}
    conds = list(dict.fromkeys(gd.condition))
    c1, c2 = st.columns(2)
    with c1:
        show(viz.grouped_bars(gd, "condition", "variant", "conflicts_per_1000",
                              "conflicts / 1,000 veh", order_x=conds, order_g=gvars, colors=gcol,
                              title="Expected conflicts", height=380), key="x-g1")
    with c2:
        show(viz.grouped_bars(gd, "condition", "variant", "avg_delay_s", "s/veh",
                              order_x=conds, order_g=gvars, colors=gcol,
                              title="Average delay", height=380), key="x-g2")
    pick = st.selectbox("Trade-off view", conds, index=conds.index(stc), key="x-g-c")
    show(viz.frontier(gd[gd.condition == pick], "avg_delay_s", "conflicts_per_1000", "variant",
                      "Average delay (s/veh)", "Expected conflicts per 1,000 veh", colors=gcol,
                      title=f"Delay vs safety, {pick} (lower-left is better)", height=380),
         key="x-g3")
    st.caption("Try it live: Simulation Lab → Smart-control add-ons → Pedestrian max-wait guard.")

    # ---- 6. Sensing uncertainty (phones per person, missed vehicles, noise) ----------
    xs, xss = T.get("x_sense"), T.get("x_sense_sum")
    if xs is not None and xss is not None:
        nb_ = xss[xss.policy != "Fixed-time"].drop_duplicates(["scenario", "bias", "noise"])
        _sp = xss[(xss.policy != "Fixed-time") & xss.bias.between(0.75, 1.5)] \
            .groupby("policy").delay_change_pct.mean()
        _ev = xss[(xss.scenario == "event_surge") & (xss.policy == "Q-learning")
                  & (xss.bias == 2.0) & (~xss.noise)].avg_delay_s.iloc[0]
        _ev0 = xss[(xss.scenario == "event_surge") & (xss.policy == "Q-learning")
                   & (xss.bias == 1.0) & (~xss.noise)].avg_delay_s.iloc[0]
        _eva = xss[(xss.scenario == "event_surge") & (xss.policy == "Actuated")
                   & (xss.bias == 2.0) & (~xss.noise)].avg_delay_s.iloc[0]
        _gap(6, "Sensing uncertainty: what if the counts are wrong (phones per person)?",
             "Every other result gives the controllers exact queue counts. Real counts come from "
             "detectors or phone/probe data: counting phones and assuming one per person "
             "over-counts when people carry two (likelihood bias above 1), low coverage misses "
             "vehicles (below 1), and sampling adds noise. The deployed agents were re-run with "
             "counts biased from 0.5× to 2× and with Poisson noise, on the 8 test episodes of every "
             "scenario (fixed-time uses no detectors and is the unaffected reference). Within "
             f"0.75-1.5× the delay changes by {_sp.get('Actuated', 0):+.1f} % for actuated, "
             f"{_sp.get('Q-learning', 0):+.1f} % for Q-learning and {_sp.get('SARSA', 0):+.1f} % for "
             f"SARSA on average. The worst case is the event surge with a 2× over-count: Q-learning "
             f"{_ev0:.1f} → {_ev:.1f} s, still ahead of actuated ({_eva:.1f} s). Whether an RL "
             f"package beats the actuated package is unchanged in {100 * nb_.same_verdict.mean():.0f} % "
             f"of the {len(nb_)} scenario × sensing conditions; the flips are in calm scenarios "
             "where the two were already within a second. Reading: the conclusions survive "
             "realistic sensing error, but RL trained on exact counts is more sensitive to it than "
             "the simple actuated rule, so a deployment would calibrate the counts (or train on "
             "noisy ones).", good=True)
        c1, c2 = st.columns([1.4, 1])
        with c1:
            pk = st.selectbox("Scenario", C.SCENARIO_ORDER, index=5, format_func=lambda k: LABEL[k],
                              key="x-sense-sc")
            nz = st.toggle("With random counting noise", key="x-sense-nz")
            d_ = xs[(xs.scenario == pk) & ((xs.noise == nz) | (xs.policy == "Fixed-time"))].copy()
            ft = d_[d_.policy == "Fixed-time"]
            d_ = pd.concat([d_[d_.policy != "Fixed-time"]]
                           + [ft.assign(bias=b) for b in sorted(xs.bias.unique())])
            show(viz.sweep_lines(d_.rename(columns={"bias": "x"}),
                                 "Queue-count bias (1 = exact; 2 = two phones per person)",
                                 "avg_delay_s", "Average delay (s/veh)", current=1.0,
                                 title=f"Delay as the counts get worse, {LABEL[pk]} "
                                       "(band = range over 8 test episodes)", height=380),
                 key="x-sense1")
        with c2:
            st.markdown("**Delay change vs exact counts (%)**, mean over scenarios")
            tb_ = xss[xss.policy != "Fixed-time"].pivot_table(
                index=["bias", "noise"], columns="policy", values="delay_change_pct").round(1)
            tb_ = tb_.reset_index()
            tb_["noise"] = tb_.noise.map({True: "Poisson", False: "none"})
            table(tb_.rename(columns={"bias": "Bias", "noise": "Noise"}), height=330)
            st.caption("Try it live: Simulation Lab → Sensing uncertainty. Related evidence: the "
                       "value of information on the decision network is about zero, and the "
                       "decision network's probabilities are Laplace-smoothed estimates learned "
                       "from simulation, not measured frequencies.")
