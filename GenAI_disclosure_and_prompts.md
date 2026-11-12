# GenAI disclosure and prompt log — Group 2 project

The unit's group-project brief (Exploration & Exploitation deck, slide 37) says:
*"If using GenAI then encircle the GenAI generated content and share the prompt separately to
differentiate your own vs GenAI content."* This file is the separate prompt record.

**Tool:** Claude (Anthropic), used through the Claude app with access to the RDMU course folder.

**How the GenAI content is marked ("encircled")**

* **Presentation** (`presentation/Smart_Traffic_Group2.pptx`): every slide with AI-generated
  content has an **amber dashed frame** around that content and the tag *"GenAI-generated
  (Claude) · prompts in log"*. Slide 16 explains the marking.
* **Requirement specification** (`Requirement_Specification.pdf`, v2.1): every AI-drafted
  section sits inside an **amber dashed frame** with a GenAI tag; the team table is the group's.
* **Code, notebook, data, dashboard**: AI-generated as listed in the table at the end.

---

## Prompt log (in date order)

Prompts marked **verbatim** are copied exactly. Where a row says **paste the verbatim prompt**,
the session's exact wording is not in this folder: copy it from that Claude chat (the AI cannot
see earlier chats) before submitting, so the log is complete.

| # | Date | Purpose | Prompt | Output |
|---|---|---|---|---|
| P1 | 24 Oct 2026 | Build the project | verbatim, below | simulator, agents, decision network, experiments, data, notebook, first dashboard, spec draft |
| P2 | 24 Oct 2026 | Answers to the AI's scoping questions | verbatim, below | scope: code + data, dashboard, spec; Streamlit |
| P3 | 24 Oct 2026 | Continue | verbatim, below | rest of the 24 Oct build |
| P4 | 7–8 Nov 2026 | Requirement specification as a Claude doc (if that version was used) | **paste the verbatim prompt** | Claude doc "Requirement Specification – Smart City Traffic Management (Group 2)" |
| P5 | Nov 2026 | Pedestrian trade-off and value of time | **paste the verbatim prompt** | `ped_tradeoff.py`, `value_of_time.py`, `make_figures.py`, `figures/`, two KPI-view sections |
| P6 | Nov 2026 | Corrections wording for the spec and slides | **paste the verbatim prompt** | `CORRECTIONS_FOR_SPEC_AND_SLIDES.md` |
| P7 | 8 Nov 2026 | Dashboard upgrade | verbatim, below | command-centre theme, Simulation Lab, linked sliders |
| P8 | 8 Nov 2026 | Research extensions and the deck | verbatim, below | `extensions.py`, `data/ext_*.csv`, Research Extensions view, 16-slide deck |
| P9 | 8 Nov 2026 | Project report and spec v2.0 PDF | **paste the verbatim prompt** | `report/Smart_Traffic_Project_Report.pdf`, `Requirement_Specification.pdf` v2.0 |
| P10 | 10 Nov 2026 | Submission review (Claude doc "Group 2 Submission Review – Smart City Traffic") | **paste the verbatim prompt** | review document listing eight gaps |
| P11 | 12 Nov 2026 | Close the review's gaps | verbatim, below | the changes listed under P11 |
| P12 | 12 Nov 2026 | Names and push | verbatim, below | names added; pushed to GitHub |
| P13 | 12 Nov 2026 | Ownership by dashboard view | verbatim, below | slide 3 and the spec's team table reorganised by view |

### P1 (24 Oct, verbatim)
> There is a folder in this called group project I need you to go through it and help me build it
> first read it, understand it - there is a more content in the RDMU folder attached to get better context

### P2 (24 Oct, answers to the AI's two scoping questions)
- *Which parts do you want me to build?* → **Code + data, Dashboard, Requirement spec** (the presentation deck was left to the group at that point).
- *What should the dashboard run on?* → **Streamlit app**.

### P3 (24 Oct, verbatim)
> continue

### P7 (8 Nov, verbatim)
> See if there are any caveats in the dashboard, fix them, add additional functionality if possible which is around the problem statement for the project. Also make the dashboard look amazing, it should look as if it's made to attract attention, the simulators and all, the live traffic and everything it shouldn't look like a demo project. Moreover, for the simulators, and what-if if there is any correlation possible in between the sliders add them too accordingly.

### P8 (8 Nov, verbatim)
> Can we fill these 5 gaps instead of making it a weakness for our project? Also for the final dashboard after making changes, I would need PPT accordingly

### P11 (12 Nov, verbatim; attached: `smart_traffic_final.zip` and `Group_2_Submission_Review___Smart_City_Traffic.pdf`)
> Read the whole thing check which gaps are needed to fill then provide the final zip with everything covered.

What P11 changed:
- **Colab link** (open to anyone, opens the notebook straight from the public GitHub repository):
  https://colab.research.google.com/github/Sbot-c/Smart_Traffic_U/blob/main/Smart_Traffic_MARL_DecisionNetwork.ipynb
- **Deck**: hosted dashboard and Colab links on the last slide (no "streamlit run"), wake-button
  note; team-and-roles slide and a lead per part on slide 2; GenAI frames
  and a GenAI slide; "LIVE" → "LIVE SIMULATION"; 280 + 80 episodes (not 360); both
  decision-network counts explained; reference article cited; new uncertainty slide; Simulation
  Lab and pedestrian guard moved to backup; value of time and AHP/TOPSIS out of the main talk;
  fresh City Overview screenshot.
- **Sensing uncertainty** (the professor's phones-per-person point): `sensing.py`,
  `data/ext_sensing*.csv`, a sensing switch in `traffic_env.py` (off by default; all published
  results unchanged, checked), Simulation Lab controls, a sixth study in Research Extensions,
  notebook section 6b (executed).
- **Dashboard**: "Live simulation · replay of simulated data" header; KPI tiles no longer cut
  off in a ~1,100 px window; project links in the sidebar; 280 + 80 wording; Policy Comparison
  headline aligned with the tuned-baseline result.
- **Specification v2.1** rebuilt under five headings (problem and requirements; data and data
  sources; algorithms; required libraries; dashboard wireframe), with the reference article.
- **UI wireframe** redrawn for the eight views (`docs/ui_wireframe.html`, `docs/dashboard_ui_diagram.png`).
- **Notebook** findings brought in line with the deck; links added.

---

## What the AI produced

| Deliverable | Files | Generated by AI? |
|---|---|---|
| Simulator, agents (Q-learning / SARSA), baselines, decision network, experiment pipeline | `config.py`, `traffic_env.py`, `agents.py`, `decision_network.py`, `experiments.py`, `run_experiments.py`, `robustness.py`, `refresh_analysis.py`, `course_methods.py` | Yes — all code |
| Research extensions and sensing study | `extensions.py`, `sensing.py`, `ped_tradeoff.py`, `value_of_time.py`, `data/ext_*.csv` | Yes — code and the generated data |
| Colab notebook | `Smart_Traffic_MARL_DecisionNetwork.ipynb` (built by `build_notebook.py`, updated 12 Nov) | Yes — code and explanatory text |
| Data CSV files | `data/*.csv` | Generated by running the AI-written simulator |
| Streamlit dashboard | `app.py`, `viz.py`, `sim_view.py`, `lab.py`, `.streamlit/config.toml` | Yes — all code |
| Requirement specification v2.1 and UI wireframe | `Requirement_Specification.pdf`, `docs/spec_source/`, `docs/ui_wireframe.html`, `docs/dashboard_ui_diagram.png` | Yes — drafted by AI (framed in amber) |
| Presentation | `presentation/Smart_Traffic_Group2.pptx` | Yes — drafted by AI (framed in amber); team names from the group |
| Project report | `report/Smart_Traffic_Project_Report.pdf` | Yes — drafted by AI |

## What the group should do before submitting

- Check that each person's responsibilities on slide 3 and in the spec's team table match
  what they actually did (ownership by dashboard view; see slide 3).
- Complete the rows above marked **paste the verbatim prompt**.
- On slide 16, replace the amber line with what the group wrote or rewrote itself, and remove
  the amber frame from any slide the group rewrote in its own words.
- Be ready to explain every modelling choice marked **ASSUMPTION** in `config.py`.

### P12 (12 Nov, verbatim)
> You can push it, I have linked the github repo anyways.
> Names:
> 1. Siddal 2. Shashaank 3. Sparsh 4. Aradhana 5. Shirley

What P12 changed: team names on slides 1-3 and in the spec; this folder pushed to the GitHub repository (live app and Colab link updated).

### P13 (12 Nov, verbatim)
> We have taken responsibility from dashboard tabs itself:
> 1. City Overview - 2. RL Simulation - Shashaank 3. Decision Network Visualizer - Siddal 4. Scenario Explorer - 5. KPI & Tradeoff Analysis - Shirley 6. Policy Comparison 7. Simulation Lab 8. Research Extensions - Shashaank
> The ones empty you can fill randomly with Sparsh and Aradhana

What P13 changed: the AI assigned the empty views (City Overview and Policy Comparison to Sparsh; Scenario Explorer and Simulation Lab to Aradhana) and rebuilt slide 3, slide 2 and the spec's team table around the eight views.
