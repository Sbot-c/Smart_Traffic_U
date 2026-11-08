# Smart City Traffic Management — Multi-Agent RL + Decision Networks

Group 2 project for *Reasoning and Decision Making under Uncertainty* (MAIB DSC 103).

Traffic lights, autonomous vehicles and buses on a 3 × 3 Downtown-Dubai-style grid learn
with **Q-learning** and **SARSA**; a **decision network (influence diagram)** learned from
the simulator scores efficiency, safety and emissions under uncertainty; six scenario
families are compared; a six-tab **Streamlit dashboard** shows everything.

## Deliverables (Blackboard, 8 Nov 2026, 10 pm)

| Deliverable | Where |
|---|---|
| Requirement specification (5 pages) | Claude doc *Requirement Specification – Smart City Traffic Management (Group 2)*, <https://claude.ai/artifact/NcNxM9NPofCuKRQMF5Km59> — export it as **PDF, A4** |
| Data CSV file | `data/smart_traffic_simulation_log.csv` (main file; the other CSVs support it) |
| Colab code link | upload `Smart_Traffic_MARL_DecisionNetwork.ipynb` to Google Drive, open with Colab, share *Anyone with the link* |
| Dashboard | `app.py` (run locally or deploy, below) |
| PPT | `presentation/Smart_Traffic_Group2.pptx` (16 slides with speaker notes) |
| GenAI prompts | `GenAI_disclosure_and_prompts.md` |

## Headline result (read this before quoting numbers)

The four dashboard "policies" are **packages**: *Actuated* = actuated signals + static routes; *Q-learning* / *SARSA* = RL signals + cooperative AV routing. Compare signal controllers only with the **same routing**:

| Rush hour (s/veh) | Static routes | Cooperative routing |
|---|---|---|
| Fixed-time signals | 109.3 | 92.1 |
| Actuated signals | 87.5 | 66.9 |
| Q-learning signals | 77.6 | 64.8 |

* Cooperative routing is the biggest lever (−16 % under fixed-time, −24 % under actuated signals).
* With the same routing, Q-learning vs actuated signals in rush hour is 64.8 vs 66.9 s (−3 %, paired p = 0.08, not significant). In the event surge Q-learning wins clearly: 46.6 vs 62.9 s (−26 %, p < 0.001).
* The extra rush-hour training made the policy worse (training delay rises 72 → 82 s). The deployed table is chosen between *general* and *scenario-adapted* on separate **validation** episodes (`data/model_selection.csv`); test episodes are never used to choose.
* Caveats: RL agents had 360 training episodes while the actuated rule was not tuned; "safety" is a stand-in based on expected conflicts, not crashes; the simulator models AV routing and platooning only (no speed control or lane selection), assumes a 25 % AV share, and cooperative routing needs vehicle-to-infrastructure communication.

| Agent | State | Actions | Reward |
|---|---|---|---|
| Traffic light (×9) | phase, green time, queue on green / red, arrivals on green, pedestrian / spillback flag (2,250 states) | keep / change phase every 5 s | −(vehicle delay + 0.5 × pedestrian delay + 300 × expected conflicts + 0.2 × g CO₂) / 100 |
| Autonomous vehicle (fleet) | remaining moves, congestion on the two candidate links (64) | next link along x or y | −travel time − cooperative congestion externality |
| Bus (fleet) | schedule deviation, load, headway, signal ahead, cross queue (144) | go / request priority / hold 20 s | −(passenger-minutes + schedule deviation + priority externality) |

## Files

```
app.py                  Streamlit dashboard (6 brief tabs + Simulation Lab), dark command-centre theme
viz.py                  Plotly figures
sim_view.py             animated night-mode canvas replay (Play / Pause / speed / scrubber, side-by-side,
                        queue sparkline, emergency-preemption badge); top-down car, autonomous-car, truck,
                        bus and ambulance sprites with headlights / brake lights (sizes exaggerated for
                        visibility, queues squeeze when they exceed the link)
extensions.py           research extensions closing five review gaps -> data/ext_*.csv (~4 min)
lab.py                  Simulation Lab: conditions -> simulator overrides, live runs, replays, sweeps,
                        correlated-condition rules, decision-network evidence mapping
config.py               every parameter, scenario definition and source
traffic_env.py          the simulator (vehicles, queues, signals, pedestrians, buses, EVs, weather)
agents.py               tabular Q-learning / SARSA, the three agent types, Webster and actuated baselines
decision_network.py     influence-diagram engine, CPT learning, expected utility, VOI, sensitivity
experiments.py          training pipeline, evaluation, component studies, data export
run_experiments.py      regenerate ./data (full ~8 min, --quick ~2 min)
robustness.py           re-adapt with different seeds -> data/training_robustness.csv
course_methods.py       uncertainty criteria, risk-averse utility, MDP table, price of anarchy, AHP, TOPSIS
refresh_analysis.py     validation-based model selection + re-evaluation from the saved Q-tables
build_notebook.py       rebuild the Colab notebook from these files
Smart_Traffic_MARL_DecisionNetwork.ipynb   the Colab notebook (code + results)
data/                   CSV outputs, trained Q-tables, animation recordings
```

## Data (`data/`)

| File | Rows | Content |
|---|---|---|
| `smart_traffic_simulation_log.csv` | 15,120 | **main dataset** — one row per junction per 30 s window, every scenario × policy |
| `kpi_results.csv` | 224 | KPIs per evaluation episode (7 scenarios × 4 policies × 8 seeds) |
| `decision_network_samples.csv` | 11,664 | junction × 10-minute samples used to learn the decision-network CPTs |
| `learning_curves.csv` | 1,680 | per-episode training metrics |
| `ablation_results.csv` | 400 | component studies: AV routing, bus priority, and all eight signal × routing pairs |
| `agent_decisions.csv` | 7,225 | recorded bus and AV decisions with their states |
| `model_selection.csv` | 14 | general vs adapted tables on validation episodes, with test results for reporting |
| `training_robustness.csv` | 18 | re-adaptation with three other random seeds |
| `scenario_parameters.csv` | 7 | scenario definitions |
| `q_tables.npz` | — | trained Q-tables (general + per-scenario) |
| `animations.json.gz` | — | vehicle trajectories for the replay |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

To regenerate every result: `python run_experiments.py` (then `python robustness.py`). To redo only the selection and evaluation from the saved Q-tables: `python refresh_analysis.py` (about 2 minutes).

## Deploy (Streamlit Community Cloud)

1. Push this folder to a **public GitHub repository** (these files at the repo root, including `data/`).
2. Go to <https://share.streamlit.io>, sign in with GitHub, click **Create app**.
3. Pick the repository, branch `main`, main file `app.py`, then **Deploy**.

## Google Colab

Upload `Smart_Traffic_MARL_DecisionNetwork.ipynb` to Google Drive, open it with Colab, and
share it with *Anyone with the link → Viewer*. *Runtime → Run all* reproduces the results.

## Added: pedestrian trade-off and value of time

| File | Purpose |
|---|---|
| `ped_tradeoff.py` | retrains rush hour at pedestrian weights 0-8 (3 training seeds each) -> `data/ped_weight_tradeoff.csv` (Q-learning only; run with `--sarsa` to add SARSA, ~20 min more) |
| `value_of_time.py` | vehicle-hours and pedestrian-hours -> AED (USDOT value of time, USD 14.10/h x 3.6725; US value, editable) |
| `make_figures.py` | slide-ready PNGs in `figures/` |
| `app.py` | two new sections at the bottom of *KPI & Trade-off Analysis* (loaded and exercised headlessly with Streamlit's AppTest; every view runs without errors) |

Slide wording for these additions is in `CORRECTIONS_FOR_SPEC_AND_SLIDES.md`.

Not done: the SARSA sweep (run `python ped_tradeoff.py --sarsa` to add it) and the Colab notebook, which was not rebuilt because that would erase its saved outputs. Note that retraining at the default pedestrian weight 0.5 gives about 69 s rush-hour delay (three runs: 64.8, 72.4, 70.1 s), not the headline 64.8 s: the difference is training randomness.

## Added: dashboard upgrade (8 Nov 2026)

**Look.** Dark "traffic command centre" theme (`.streamlit/config.toml`, CSS in `app.py`, palette in
`viz.py`): hero header with live headline figures, a ticker of every scenario's best package, a
segmented view switcher, glass KPI cards, and a night-mode city replay with lit towers, glowing signal
heads, headlights, a cars-queued sparkline with playhead and an *EV PREEMPTION* badge.

**Simulation Lab** (replaces *Live What-If*). Eleven conditions in three groups: demand, rush-hour ramp,
pedestrians, bus passengers; weather, crash on Street 2 with capacity loss, arena event, emergency calls;
AV share, bus lane, bus headway. Every change re-runs all four policies on identical arrivals
(2-8 seeds, ~0.06 s per episode). It shows:
* score per package under the sidebar weights, next to the decision network's recommendation for the
  same conditions (conditions mapped to Weather / Traffic density / Pedestrian flow evidence);
* seven criteria side by side (delay, safety, CO2, NOx, pedestrian wait, bus on time, throughput) and
  emergency-vehicle trips and delay;
* a side-by-side replay of these exact conditions;
* a stress test that sweeps demand, pedestrians, AV share or crash severity for every policy;
* a correlation map of the linked conditions.

**Linked (correlated) sliders.** On by default; switch off to move each condition freely.
| Driver | Moves | Rule (from the scenario definitions in `config.py`) |
|---|---|---|
| Rush-hour ramp | demand, pedestrians | 1.30 × normal |
| Demand | pedestrians | 1 : 1 |
| Demand | bus passengers | demand ^ 2.24 (rush hour 1.30 → 1.8) |
| Event surge | bus passengers | × 1.5 |
| Weather | emergency calls | rain / sandstorm 4 per 30 min instead of 3 |
| Crash | emergency calls | + 3 |
| Bus lane | bus headway | 3 minutes |

Weather's effect on demand, pedestrians, capacity, speed and crash risk is applied inside the simulator
in both modes and shown as "effective" chips. The stakeholder weights in the sidebar are linked too:
they are shares of one budget, so raising one lowers the other two in proportion (toggle to unlink).
These links are planning assumptions taken from the scenario settings, not measured elasticities.

**Fixes.** What-if chart no longer labels custom conditions "Normal flow"; navigation styling works on
Streamlit 1.67; all-zero weights fall back to the defaults with a warning; the empty slot in *KPI &
Trade-off Analysis* now holds NOx, plus idle-time and throughput heatmaps; *Policy Comparison* adds NOx,
idle time, stops, emergency-vehicle delay and share of pedestrians waiting > 30 s; the
uncertainty-criteria caption is computed from the data; replay labels no longer wrap.

**Simulator change.** `traffic_env.py` reads an optional `av_share` override (default `config.AV_SHARE`),
so every published result is unchanged (checked: identical what-if KPIs before and after).

## Added: research extensions (8 Nov 2026, `extensions.py`, dashboard view *Research Extensions*)

Five gaps from the project review, each closed with an experiment on the deployed Q-tables (no
retraining), the 8 held-out test episodes for reporting and validation episodes for any tuning.
The new mechanisms are scenario overrides that default to off, so every other result is unchanged
(checked: rush-hour Q-learning seed 0 still gives 61.6912 s).

| Gap | What was added | Result |
|---|---|---|
| 1. AV speed control / lane selection | V2I speed advisory (AVs ease off towards a red light, max 15 s per link) and AV lane selection (AVs pass a vehicle waiting to turn or blocked by a full exit) | lane selection: AV delay −2.2 %, network delay −0.7 %; speed advisory: stops −0.4 %, conflicts −1.1 %, CO2 −0.2 %, AV delay +6.8 % (means over all scenarios, 25 % AVs) |
| 2. Bus load balancing | load KPIs (load at each stop, spread, 95th percentile, passengers left behind) and an even-headway holding rule | rush hour: busiest buses full and ~35 passengers per window left behind under every holding strategy: capacity, not balance; a bus every 3 min cuts it to ~1. RL bus agents keep the best on-time record (66 % vs 57 %) |
| 3. Proxy safety model | each conflict mechanism halved and doubled, all policies re-run | safest policy unchanged in 90 % of 42 cases; changes only swap the two non-learning controllers |
| 4. Fair comparison | actuated controller tuned per scenario on validation runs (81 settings); RL averaged over training runs | same cooperative routing: event surge 46.6 (Q) vs 46.9 s (tuned actuated), a tie; rush hour 64.8 vs 57.7 s, tuned actuated better. Q-learning re-trained ×3: rush hour 72.5 ± 2.1 s |
| 5. RL safety at high demand | pedestrian max-wait guard (safety supervisor forces service after 22-30 s) | stress (1.6 × demand, peak): conflicts 18.6 → 10.7 per 1,000 veh (actuated package 10.6) with delay 179 vs 205 s; costs delay in rush hour (64.8 → 76.2 s) and is counter-productive in the event surge |

The Simulation Lab has three matching switches (speed advisory, lane selection, pedestrian guard).

**What this changes in the story:** the earlier "RL signals win the event surge by 26 %" was against an
untuned actuated controller. Against a tuned one it is a tie, and in rush hour tuned actuated is better.
Cooperative AV routing remains the gain that survives a fair comparison. See
`CORRECTIONS_FOR_SPEC_AND_SLIDES.md` for the wording.
