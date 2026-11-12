"""Requirement specification v2.1 (Group 2, Smart City Traffic) - reportlab build.

Team names are in TEAM below. To rebuild, from the project folder run:
    python docs/spec_source/spec_figs.py      (figures, optional: already included)
    python docs/spec_source/build_spec.py     -> Requirement_Specification.pdf
Needs: pip install reportlab matplotlib (DejaVu fonts, standard on Linux/Colab).
"""
import os
import sys

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

_H = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_H))
OUT, DATA, WORK, WIRE = (sys.argv[1:5] if len(sys.argv) >= 5 else
                         [os.path.join(_ROOT, "Requirement_Specification.pdf"),
                          os.path.join(_ROOT, "data"), _H,
                          os.path.join(_ROOT, "docs", "dashboard_ui_diagram.png")])
F = next((d for d in ["/usr/share/fonts/truetype/dejavu/", "/usr/share/fonts/TTF/",
          os.path.join(os.path.dirname(__import__("matplotlib").__file__), "mpl-data", "fonts",
                       "ttf") + "/"] if os.path.exists(d + "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DV", F + "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DVB", F + "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DVI", F + "DejaVuSans-Oblique.ttf"))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DVB", italic="DVI", boldItalic="DVB")

INK, MUTED, BLUE, AMBER = colors.HexColor("#111111"), colors.HexColor("#5a5953"), \
    colors.HexColor("#1f5fae"), colors.HexColor("#c98a00")
LINE, HEAD = colors.HexColor("#d9d7cf"), colors.HexColor("#f2f1ec")
DASH_URL = "https://smarttrafficu-mmzjikh5bg8frby3adzdc2.streamlit.app/"
COLAB_URL = ("https://colab.research.google.com/github/Sbot-c/Smart_Traffic_U/blob/main/"
             "Smart_Traffic_MARL_DecisionNetwork.ipynb")
GH_URL = "https://github.com/Sbot-c/Smart_Traffic_U"
TEAM = ["Siddal", "Shashaank", "Sparsh", "Aradhana", "Shirley"]

ss = dict(
    title=ParagraphStyle("t", fontName="DVB", fontSize=15.5, leading=19, textColor=INK),
    sub=ParagraphStyle("s", fontName="DV", fontSize=8.3, leading=11, textColor=MUTED),
    h1=ParagraphStyle("h1", fontName="DVB", fontSize=11.5, leading=14, textColor=BLUE,
                      spaceBefore=2, spaceAfter=3),
    h2=ParagraphStyle("h2", fontName="DVB", fontSize=8.8, leading=11, textColor=INK,
                      spaceBefore=4, spaceAfter=1.5),
    p=ParagraphStyle("p", fontName="DV", fontSize=8.1, leading=10.6, textColor=INK,
                     alignment=TA_LEFT, spaceAfter=2.5),
    cell=ParagraphStyle("c", fontName="DV", fontSize=7.3, leading=9.1, textColor=INK),
    cellb=ParagraphStyle("cb", fontName="DVB", fontSize=7.3, leading=9.1, textColor=INK),
    cap=ParagraphStyle("cap", fontName="DVI", fontSize=7.2, leading=9, textColor=MUTED,
                       spaceAfter=3),
    tag=ParagraphStyle("tag", fontName="DVB", fontSize=6.6, leading=8, textColor=AMBER),
    eq=ParagraphStyle("eq", fontName="DV", fontSize=8.1, leading=11, textColor=INK, leftIndent=12,
                      spaceAfter=1),
)
W = A4[0] - 30 * mm


def P(t, st="p"):
    return Paragraph(t, ss[st])


def tbl(rows, widths, head=True, fs="cell"):
    data = [[c if not isinstance(c, str) else
             P(c, "cellb" if (head and i == 0) else fs) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * (W - 14) for w in widths], repeatRows=1 if head else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, LINE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
          ("TOPPADDING", (0, 0), (-1, -1), 1.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.8)]
    if head:
        st.append(("BACKGROUND", (0, 0), (-1, 0), HEAD))
    t.setStyle(TableStyle(st))
    return t


def framed(items, tag="GenAI-generated (Claude) · edited and checked by the group · prompts "
                      "in GenAI_disclosure_and_prompts.md"):
    """Encircle GenAI content (the brief: 'encircle the GenAI generated content')."""
    lst = lambda x: x if isinstance(x, list) else [x]
    first = [P(tag, "tag")] + lst(items[0])
    rest = items[1:]
    if not isinstance(items[0], list) and rest:      # keep a heading with what follows it
        first += lst(rest[0])
        rest = rest[1:]
    rows = [[first]] + [[it] for it in rest]
    t = Table(rows, colWidths=[W])
    t.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.9, AMBER, None, (3, 2)),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 0.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 0.6),
        ("TOPPADDING", (0, 0), (0, 0), 3)]))
    return t


def img(path, width):
    im = Image(path)
    r = im.imageHeight / im.imageWidth
    im.drawWidth, im.drawHeight = width, width * r
    return im


# --------------------------------------------------------------------------------------
# numbers from the data (so the document and the dashboard agree)
# --------------------------------------------------------------------------------------
xs = pd.read_csv(f"{DATA}/ext_sensing_summary.csv")
nb = xs[xs.policy != "Fixed-time"].drop_duplicates(["scenario", "bias", "noise"])
same = 100 * nb.same_verdict.mean()
rng = xs[(xs.policy != "Fixed-time") & xs.bias.between(0.75, 1.5)].groupby("policy") \
    .delay_change_pct.mean()
ev = xs[(xs.scenario == "event_surge") & (~xs.noise)]
q0 = ev[(ev.policy == "Q-learning") & (ev.bias == 1.0)].avg_delay_s.iloc[0]
q2 = ev[(ev.policy == "Q-learning") & (ev.bias == 2.0)].avg_delay_s.iloc[0]
a2 = ev[(ev.policy == "Actuated") & (ev.bias == 2.0)].avg_delay_s.iloc[0]

story = []
story += [P("Requirement Specification: Smart City Traffic Management", "title"), Spacer(1, 2),
          P("Group 2 · Reasoning and Decision Making under Uncertainty (MAIB DSC 103) · "
            "Version 2.1, 12 November 2026 (replaces v2.0 of 8 November)", "sub"),
          P(f'Live dashboard (live simulation, no installation): <link href="{DASH_URL}" '
            f'color="#1f5fae">{DASH_URL}</link> · Colab code (open to anyone with the link): '
            f'<link href="{COLAB_URL}" color="#1f5fae">colab.research.google.com/github/'
            f'Sbot-c/Smart_Traffic_U/…/Smart_Traffic_MARL_DecisionNetwork.ipynb</link> · '
            f'Code and data: <link href="{GH_URL}" color="#1f5fae">github.com/Sbot-c/'
            f'Smart_Traffic_U</link>', "sub"), Spacer(1, 4)]

story.append(tbl([["Part of the brief", "Lead", "Responsible for"],
                  ["1 Multi-agent system", TEAM[0], "agent design (states, actions, rewards), "
                   "simulator and safety shield"],
                  ["2 Reinforcement learning", TEAM[1], "Q-learning and SARSA training, model "
                   "selection, tuned baseline"],
                  ["3 Decision network", TEAM[2], "influence diagram, expected utility, value of "
                   "information, sensing uncertainty"],
                  ["4 Scenario experiments", TEAM[3], "seven scenarios, KPIs, statistics, AV and "
                   "bus studies, data CSV"],
                  ["5 Command dashboard", TEAM[4], "Streamlit dashboard, replay, Simulation Lab, "
                   "deployment"]], [0.25, 0.2, 0.55]))
story.append(P("Team table written by the group. Every section inside "
               "an amber dashed frame was drafted with GenAI and then checked and edited by the "
               "group; the prompts are in <i>GenAI_disclosure_and_prompts.md</i>.", "cap"))

# ======================================================================================
# 1. Problem statement, objectives and requirements
# ======================================================================================
s1 = [P("1. Problem statement, objectives and requirements", "h1"),
      P("<b>Problem.</b> A city traffic authority must choose how to control a signalised "
        "network when demand, weather, pedestrians, incidents and emergencies are uncertain, "
        "and must trade off efficiency, safety and emissions. We specify a decision-support "
        "system for a 3 × 3 Downtown-Dubai-style grid (nine junctions, 400 m blocks, a "
        "three-lane eastbound corridor on Street 2) in which traffic lights, autonomous "
        "vehicles (AVs) and buses are learning agents (Q-learning and SARSA), a decision network "
        "chooses the control strategy, and a dashboard shows the evidence as a live simulation "
        "of simulated data. It is a research and teaching prototype on synthetic demand, not a "
        "deployed controller. The starting point is Wairagade, Ugale &amp; Waghmare (2025), who "
        "propose RL signal timing with an emergency override and a monitoring dashboard but "
        "report no quantitative evaluation; this project builds and evaluates such a system."),
      P("<b>Stakeholders.</b> Road-authority operators (choose the strategy), commuters and AV "
        "operators (travel time), bus operators and passengers (punctuality, crowding), "
        "pedestrians (waiting, safety), emergency services (pre-emption), residents (emissions)."),
      tbl([["ID", "Objective", "Measured by (KPI)"],
           ["O1", "Reduce congestion and delay", "average delay (s/veh), throughput (veh/h), stops"],
           ["O2", "Improve safety for vehicles and pedestrians",
            "expected conflicts per 1,000 veh, jaywalks, pedestrian wait"],
           ["O3", "Lower emissions", "CO₂ (kg), NOx (g), idle time; health-impact node"],
           ["O4", "Keep public transport reliable", "bus on-time %, bus delay, load, left behind"],
           ["O5", "Decide well under uncertainty",
            "expected utility, value of information, weight and sensing sensitivity"]],
          [0.06, 0.36, 0.58]),
      [P("Functional requirements", "h2"), tbl([["ID", "The system shall …", "Implemented by"],
           ["FR1", "simulate a signalised grid with time-varying demand, turns, queues, "
            "spill-back and lost time", "traffic_env.py: 5 s steps, 30-min episodes, 5-min warm-up"],
           ["FR2", "model peak demand, pedestrian crossings and jaywalking, emergency-vehicle "
            "pre-emption, rain, sandstorm, incidents and an arena event",
            "scenario parameters (below); pedestrians and emergency vehicles run in every scenario"],
           ["FR3", "let traffic-light agents keep or change phase", "Q-learning / SARSA, one "
            "table per junction (2,250 states)"],
           ["FR4", "let AV agents route cooperatively, with speed control and lane selection",
            "RL routing (64 states); V2I speed advisory; lane selection"],
           ["FR5", "let bus agents request priority or hold; measure loads", "RL bus agents "
            "(144 states): go / priority / hold 20 s; load KPIs"],
           ["FR6", "keep every controller inside a safety shield", "min/max green, pedestrian "
            "clearance, EV pre-emption, bus priority; optional pedestrian max-wait guard"],
           ["FR7", "compare against fixed-time and actuated baselines", "Webster plan; actuated "
            "rule; actuated tuned on validation runs; signal × routing ablation"],
           ["FR8", "evaluate strategies with a decision network", "chance, decision and utility "
            "nodes; expected utility; value of information (section 3.5)"],
           ["FR9", "report every KPI with its uncertainty", "8 held-out test episodes, mean ± "
            "s.d., paired tests; robustness studies"],
           ["FR10", "provide an interactive dashboard and export the data", "Streamlit, 8 views "
            "(section 5); CSV download"]], [0.06, 0.52, 0.42])],
      [P("Non-functional requirements", "h2"), tbl([["ID", "Requirement", "Target / evidence"],
           ["NFR1", "Reproducibility", "fixed seeds (training; validation 800,000+; test "
            "900,000+); common random numbers; every CSV regenerated by scripts"],
           ["NFR2", "Honest evaluation", "choices made on validation runs only; test episodes "
            "never used to choose; spread over three re-trainings reported"],
           ["NFR3", "Performance", "≈0.06-0.17 s per 30-min episode on a CPU; full pipeline in "
            "minutes; the Lab re-runs four policies in about a second"],
           ["NFR4", "Explainability", "tabular Q-values, logged decisions and shield overrides, "
            "readable probabilities"],
           ["NFR5", "Usability and access", "hosted link, no installation; plain-language "
            "captions; best viewed full screen; tiles wrap in narrow windows"]],
          [0.07, 0.2, 0.73])],
      [P("Scenarios (all 30 simulated minutes; pedestrians and emergency vehicles in each)", "h2"),
      tbl([["Scenario", "Key parameters"],
           ["Normal flow", "demand 1.0, clear, bus every 6 min, 3 emergency calls / 30 min"],
           ["Rush hour", "demand 1.3 ramping to ≈1.5, eastbound corridor +15 %, pedestrians "
            "1.3, bus passengers 1.8"],
           ["Incident", "crash before J5 eastbound, minutes 5-25: capacity −75 %, speed −60 %, "
            "6 emergency calls"],
           ["Rain / Sandstorm", "capacity −6 % / −12 %, speed −10 % / −20 %, crash risk ×1.71 / "
            "×2.0, demand ×0.95 / ×0.85, pedestrians ×0.60 / ×0.30, 4 emergency calls"],
           ["Event surge", "arena lets out at minute 8: cars and pedestrians surge at three "
            "junctions, bus passengers ×1.5"],
           ["Bus corridor", "dedicated bus lane on Street 2, bus every 3 min, passengers ×2.0"]],
          [0.18, 0.82])]]
story.append(framed(s1))
story.append(Spacer(1, 6))

# ======================================================================================
# 2. Data and data sources
# ======================================================================================
s2 = [P("2. Data and data sources", "h1"),
      P("All traffic data are <b>generated by the simulator</b> (there is no public, "
        "junction-level Dubai feed for these conditions); the parameters that drive it come "
        "from the published sources below, and the values marked <i>assumption</i> are our "
        "modelling choices (listed in <i>config.py</i>)."),
      tbl([["Data file (data/)", "Rows", "Content"],
           ["smart_traffic_simulation_log.csv (main)", "15,120", "one row per junction per "
            "30 s window, every scenario × policy: queues, arrivals, departures, delay, "
            "pedestrians, jaywalks, conflicts, CO₂, NOx, stops, spill-back, EV and bus flags, reward"],
           ["kpi_results.csv", "224", "KPIs per test episode (7 scenarios × 4 policies × 8)"],
           ["decision_network_samples.csv", "11,664", "junction × 10-min samples used to learn "
            "the decision-network probabilities"],
           ["learning_curves.csv · model_selection.csv", "1,680 · 14", "training metrics; "
            "general vs adapted tables on validation runs"],
           ["ablation_results.csv · training_robustness.csv", "400 · 18", "signal × routing and "
            "component studies; re-training with other seeds"],
           ["ext_*.csv (incl. ext_sensing*.csv)", "—", "research extensions and the sensing-"
            "uncertainty study (2,072 episodes)"],
           ["q_tables.npz · animations.json.gz", "—", "trained Q-tables; replay trajectories"]],
          [0.36, 0.1, 0.54]),
      Spacer(1, 3),
      tbl([["Parameter", "Value used", "Source"],
           ["Network layout", "3 × 3 grid, ≈400 m blocks", "Downtown Dubai / Business Bay "
            "street grid (approximation)"],
           ["Saturation flow, lost time, pedestrian non-compliance", "1,800 veh/h/lane; "
            "patience 30 s", "Transportation Research Board (2016), Highway Capacity Manual, "
            "6th ed."],
           ["Fixed-time plan", "Webster optimum cycle", "Webster (1958), Road Research Technical "
            "Paper 39"],
           ["Rain: capacity, speed", "−6 %, −10 %", "Perrin et al. (2002) in Sullivan, "
            "Sisiopiku &amp; Reddy-Surhabi (2010), UTCA 07211"],
           ["Rain: crash rate", "+71 %", "Qiu &amp; Nixon (2008), Transportation Research "
            "Record 2055"],
           ["Sandstorm (low visibility): capacity", "−12 %", "HCM via FHWA (2013), FHWA-HOP-13-042; "
            "speed −20 % and crash ×2.0 are assumptions"],
           ["CO₂ and idling", "400 g CO₂/mile; 0.3 / 1 gal/h idle", "US EPA (2023); Argonne "
            "National Laboratory idling fact sheet"],
           ["Value of time", "USD 14.10 /person-h = AED 51.8", "U.S. DOT value of travel time "
            "(US figure, editable)"],
           ["AV share, platooning, red-light running", "25 %; 0.85 headway; 8 %", "assumptions"]],
          [0.3, 0.22, 0.48])]
story.append(framed(s2))
story.append(Spacer(1, 6))

# ======================================================================================
# 3. Algorithms
# ======================================================================================
s3 = [P("3. Algorithms and methods", "h1"),
      [P("3.1 Agents (each with its own state, actions and reward)", "h2"), tbl([["Agent", "State (discretised)", "Actions", "Reward (maximised = minus cost)"],
           ["Traffic light ×9", "phase, green time, queue on green and red, arrivals on green, "
            "pedestrian / spill-back flag (2,250)", "keep / change phase every 5 s",
            "−(delay + 0.5 × pedestrian delay + 300 × expected conflicts + 0.2 × g CO₂) / 100"],
           ["Autonomous vehicle (fleet)", "remaining moves, congestion of the two candidate "
            "links (64)", "next link; V2I speed advisory; lane selection",
            "−(travel time + congestion imposed on others)"],
           ["Bus (fleet)", "schedule deviation, load, headway, signal ahead, cross queue (144)",
            "go / request priority / hold 20 s", "−(passenger-minutes + schedule deviation + "
            "priority cost to cross traffic)"]], [0.16, 0.34, 0.2, 0.3])],
      P("3.2 Learning rules (tabular, ε-greedy)", "h2"),
      P("Q-learning (off-policy): Q(s,a) ← Q(s,a) + α [ r + γ max<sub>a'</sub> Q(s',a') − Q(s,a) ]",
        "eq"),
      P("SARSA (on-policy): Q(s,a) ← Q(s,a) + α [ r + γ Q(s',a') − Q(s,a) ], with a' the action "
        "actually taken", "eq"),
      P("α = 0.10, γ = 0.90 (AV routing 0.98), ε from 0.30 to 0.02 over the first 80 % of "
        "episodes. Q starts at 0; in a state never visited in training the signal falls back "
        "to the actuated rule (at most 7 of 25,920 decisions per scenario). Every action passes "
        "the safety shield (FR6). Tables, not deep RL: 2,250 states per junction, every decision "
        "can be inspected, training takes minutes on a CPU."),
      P("3.3 Training and evaluation protocol", "h2"),
      P("<b>280 pre-training episodes</b> over all scenarios (the nine junctions share one table; "
        "weather, pedestrian level and demand are randomised so the policy is robust to them), "
        "then <b>80 adaptation episodes per scenario</b> with nine independent tables. For each "
        "scenario the general or the adapted table is chosen on 8 validation episodes; results "
        "are reported on 8 separate test episodes with the same arrivals for every policy. "
        "Comparisons use paired t-tests. Baselines: Webster fixed-time; vehicle-actuated "
        "(gap-out, queue ratio, pedestrian service, 60 s max green); the actuated rule tuned per "
        "scenario on validation runs (81 settings)."),
      P("3.4 Policy packages", "h2"),
      P("<i>Actuated</i> = actuated signals + static routes; <i>Q-learning / SARSA</i> = RL "
        "signals + cooperative AV routing. Signal controllers are compared with the same routing "
        "(signal × routing ablation), so routing gains are not credited to the signals."),
      P("3.5 Decision network (influence diagram)", "h2"),
      P("Chance nodes Weather {clear, rain, sandstorm}, Traffic density and Pedestrian flow "
        "{low, medium, high} → Congestion, Collision risk, Emissions → Health impact; decision "
        "nodes Signal timing {fixed, actuated, Q-learning, SARSA} × Routing {static, cooperative} "
        "(8 options); utility nodes efficiency, safety, environment with weights 0.40 / 0.35 / "
        "0.25 (adjustable). Conditional probabilities are learned from the 11,664 simulated "
        "samples with a Dirichlet(1) prior (Laplace smoothing); expected utility is computed by "
        "enumeration (cross-checked with pgmpy variable elimination). Outputs: the best decision "
        "per condition, value of information (EVPI) and weight sensitivity. Result: actuated "
        "signals + cooperative routing in 6 of 7 scenarios (rain: + static routes), i.e. in 22 "
        "of 27 weather × density × pedestrian conditions; these two counts measure different "
        "things."),
      P("3.6 Where the uncertainty is, and how it is handled", "h2"),
      P("(a) <b>Sensing / likelihood bias.</b> Real queue counts come from detectors or phone "
        "data; counting phones and assuming one per person over-counts when people carry two, "
        "low coverage under-counts. The deployed agents were re-run with counts biased 0.5×-2× "
        f"and with Poisson noise (2,072 test episodes, <i>sensing.py</i>). Within 0.75-1.5× the "
        f"delay changes by {rng['Actuated']:+.1f} % (actuated), {rng['Q-learning']:+.1f} % "
        f"(Q-learning) and {rng['SARSA']:+.1f} % (SARSA); worst case event surge at 2×: Q-learning "
        f"{q0:.1f} → {q2:.1f} s, still ahead of actuated ({a2:.1f} s). The RL-vs-actuated verdict "
        f"is unchanged in {same:.0f} % of {len(nb)} conditions (Figure 3). (b) <b>Value of "
        "information</b> ≤ 0.004 utility points: observing a condition first would not change "
        "the decision. (c) <b>Safety model</b>: halving or doubling each conflict mechanism keeps "
        "the safest policy in 90 % of 42 cases. (d) <b>Training randomness</b>: Q-learning "
        "re-trained three times gives 72.5 ± 2.1 s in rush hour. (e) The network's "
        "probabilities are smoothed estimates learned from simulation, not measured "
        "frequencies.")]
story.append(framed(s3))
story.append(Spacer(1, 6))

# ======================================================================================
# 4. Libraries
# ======================================================================================
s4 = [P("4. Required libraries and tools", "h1"),
      tbl([["Library / tool", "Version", "Used for"],
           ["Python", "3.11 or later", "everything"],
           ["numpy", "≥ 1.26", "simulator state, Q-tables, random streams"],
           ["pandas", "≥ 2.0", "logs, KPIs, CSV export, decision-network samples"],
           ["scipy", "≥ 1.11", "paired t-tests and statistics"],
           ["streamlit", "≥ 1.49", "dashboard (app.py); hosted on Streamlit Community Cloud"],
           ["plotly", "≥ 5.20", "dashboard charts and network map"],
           ["matplotlib", "any recent", "notebook and report figures"],
           ["pgmpy", "any recent (installed in the notebook)", "cross-check of the decision-"
            "network inference"],
           ["Google Colab · GitHub · Streamlit Community Cloud", "—", "notebook (code link), "
            "source and data, public dashboard"]], [0.36, 0.22, 0.42]),
      P("Install: <i>pip install -r requirements.txt</i>. Hardware: an ordinary CPU; no GPU.",
        "cap")]
story.append(framed(s4))
story.append(Spacer(1, 6))

# ======================================================================================
# 5. Dashboard wireframe
# ======================================================================================
s5 = [[P("5. Dashboard design (user-interface wireframe)", "h1"),
      img(WIRE, W - 14),
      P("Figure 1. Wireframe of the eight views. The sidebar (scenario, policy, linked "
        "stakeholder weights, downloads, links) drives every view; the header is labelled "
        "<i>live simulation</i> because it replays and re-runs simulated data.", "cap")],
      tbl([["View", "Must show"],
           ["City Overview", "animated replay with signals, queues and counters; KPI tiles vs "
            "fixed-time; network map with a minute slider"],
           ["RL Simulation", "side-by-side replay of two policies on the same arrivals; agent "
            "decisions; learned policy; learning curves"],
           ["Decision Network Visualizer", "influence diagram with evidence; expected utility "
            "of all 8 options; value of information; weight sensitivity; probability tables"],
           ["Scenario Explorer", "the seven scenarios, their parameters and KPI distributions"],
           ["KPI &amp; Trade-off Analysis", "improvement heatmaps; delay vs safety; course "
            "techniques (back-up material)"],
           ["Policy Comparison", "signal × routing ablation with paired statistics; model "
            "selection; Q-learning vs SARSA"],
           ["Simulation Lab", "live re-run of all policies under linked conditions, add-ons "
            "and sensing bias; replay; stress test"],
           ["Research Extensions", "six studies: AV speed and lanes, bus loads, safety proxy, "
            "tuned baseline, pedestrian guard, sensing uncertainty"]], [0.25, 0.75])]
story.append(framed(s5))
story.append(Spacer(1, 6))

# ======================================================================================
# Appendix: results and acceptance
# ======================================================================================
sa = [[P("Appendix A. Results and acceptance (8 held-out test episodes per result)", "h1"),
      img(f"{WORK}/fig_delay.png", W - 14),
      P("Figure 2. Average delay by scenario and policy package (mean ± s.d.).", "cap")],
      tbl([["Criterion", "Result"],
           ["Delay vs fixed-time (packages)", "Q-learning package −41 % in rush hour (64.8 vs "
            "109.3 s) and −49 % in the event surge (46.6 vs 90.6 s); adaptive controllers within "
            "1-2 s of each other in calm conditions"],
           ["Signals vs routing", "cooperative routing alone −24 % under actuated signals (87.5 → "
            "66.9 s); Q-learning vs actuated signals with the same routing −3 % in rush hour "
            "(p = 0.08, not significant)"],
           ["Fair baseline", "with the actuated controller tuned on validation runs and the same "
            "routing: event surge 46.6 (Q) vs 46.9 s, a tie; rush hour 64.8 vs 57.7 s, tuned "
            "actuated better. RL reaches tuned-expert level without manual tuning"],
           ["AV speed control and lanes", "lane selection: AV delay −2.2 %; speed advisory: "
            "fewer stops, conflicts −1.1 %, for +6.8 % AV delay"],
           ["Bus loads", "rush-hour buses run full under every holding strategy; a bus every 3 "
            "min cuts left-behind passengers from ≈35 to ≈1"],
           ["High-demand safety", "pedestrian max-wait guard at 1.6 × demand: conflicts 18.6 → "
            "10.7 per 1,000 veh (actuated 10.6), delay 179 vs 205 s"],
           ["Uncertainty", f"section 3.6: sensing bias (verdict kept in {same:.0f} %), VOI ≈ 0, "
            "safety-model sensitivity 90 %, re-training 72.5 ± 2.1 s"]], [0.24, 0.76]),
      Spacer(1, 3),
      img(f"{WORK}/fig_sensing.png", W - 14),
      P("Figure 3. Delay when the signal controllers see biased queue counts (no counting noise). "
        "Fixed-time uses no detectors and is unaffected.", "cap"),
      P("<b>Limitations.</b> Synthetic grid not calibrated to observed Dubai counts; 25 % AV "
        "share; expected conflicts are a crash surrogate (rankings robust, absolute values not "
        "crash rates); agents were trained on exact counts; value of time is a U.S. figure; NOx "
        "factors are order-of-magnitude; tabular RL limits state detail. <b>Next:</b> calibrate "
        "with RTA detector data, train on noisy counts, learn the speed advisory, test a larger "
        "network."),
      P("<b>Deliverables.</b> This specification; data CSV (data/smart_traffic_simulation_log.csv "
        "plus supporting files); Colab notebook (link above); hosted dashboard (link above); "
        "presentation (presentation/Smart_Traffic_Group2.pptx) and report; GenAI prompt log "
        "(GenAI_disclosure_and_prompts.md)."),
      P("<b>Reference.</b> Wairagade, A., Ugale, K. &amp; Waghmare, R. (2025). AI-Powered Smart "
        "Traffic Management System for Urban Congestion Reduction. <i>International Journal of "
        "Scientific Research &amp; Engineering Trends</i>, 11(2), 1997-2001. Further sources: "
        "Kochenderfer (2015), <i>Decision Making Under Uncertainty</i>, MIT Press; Sutton &amp; "
        "Barto (2018), <i>Reinforcement Learning</i>, 2nd ed., MIT Press; and those in section 2.")]
story.append(framed(sa))


def footer(c, d):
    c.saveState()
    c.setFont("DV", 6.8)
    c.setFillColor(MUTED)
    c.drawString(15 * mm, 9 * mm, "Group 2 · Requirement Specification · Smart City Traffic "
                 "Management · v2.1")
    c.drawRightString(A4[0] - 15 * mm, 9 * mm, str(d.page))
    c.setFillColor(AMBER)
    c.drawRightString(A4[0] - 15 * mm, A4[1] - 9 * mm,
                      "Amber dashed frame = GenAI-generated content (see prompt log)")
    c.restoreState()


doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
                        topMargin=13 * mm, bottomMargin=14 * mm,
                        title="Requirement Specification - Smart City Traffic Management (v2.1)",
                        author="Group 2")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("built", OUT)
