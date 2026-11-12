# GenAI disclosure and prompt log — Group 2 project

The unit's group-project brief (Exploration & Exploitation deck, slide 37) says:
*"If using GenAI then encircle the GenAI generated content and share the prompt separately to
differentiate your own vs GenAI content."* This file is the separate prompt record.

**Tool:** Claude (Anthropic), used through the Claude app with access to the RDMU course folder.

**How the GenAI content is marked ("encircled")**

* **Presentation** (`presentation/Smart_Traffic_Group2.pptx`): every slide with AI-generated
  content has an **amber dashed frame** around that content and the tag *"GenAI-generated
  (Claude) · prompts in log"*. Slide 16 explains the marking and lists what each of us did.
* **Requirement specification** (`Requirement_Specification.pdf`, v2.1): every AI-drafted
  section sits inside an **amber dashed frame** with a GenAI tag; the team table is the group's.
* **Code, notebook, data, dashboard**: AI-generated as listed in the table at the end.

---

## Prompt log (in date order)

The project was built in three Claude chats. Prompts marked **verbatim** are copied exactly from
the chat. Still missing: P8, P9 and the first request of P10 (see the note under Chat B).

| # | Date | Chat | Purpose | Prompt | Output |
|---|---|---|---|---|---|
| P1 | 24 Oct 2026 | A | Build the project | verbatim, below | simulator, agents, decision network, experiments, data, notebook, first dashboard |
| P2 | 24 Oct 2026 | A | Answers to the AI's two scoping questions | verbatim, below | scope: code + data, dashboard, spec; Streamlit |
| P3 | 24 Oct 2026 | A | Continue | verbatim, below | rest of the 24 Oct build, and the requirement specification drafted as a Claude doc (no separate prompt) |
| P4 | 7 Nov 2026 | A | Critique against the subject outline (section 3.2) and class decks | verbatim, below | review from several roles: fair comparison, model selection, course techniques, UI |
| P5 | 7–8 Nov 2026 | A | Apply the critique | verbatim, below | an alternative 8 Nov build and a revised Claude-doc spec (**not** the submitted build; its ideas fed the corrections in Chat B) |
| P6 | 8 Nov 2026 | B | Dashboard upgrade | verbatim, below | command-centre theme, Simulation Lab, linked sliders |
| P7 | 8 Nov 2026 | B | Research extensions and the deck | verbatim, below | `extensions.py`, `data/ext_*.csv`, Research Extensions view, 16-slide deck |
| P8 | ≈7–8 Nov 2026 | B | Pedestrian trade-off and value of time | **paste the verbatim prompt** | `ped_tradeoff.py`, `value_of_time.py`, `make_figures.py`, `figures/`, two KPI-view sections |
| P9 | ≈7–8 Nov 2026 | B | Corrections wording for the spec and slides | **paste the verbatim prompt** | `CORRECTIONS_FOR_SPEC_AND_SLIDES.md` |
| P10 | 8 Nov 2026 | B | Project report and spec v2.0 PDF | first request: **paste the verbatim prompt**; two follow-ups verbatim, below | `report/Smart_Traffic_Project_Report.pdf`, `Requirement_Specification.pdf` v2.0 |
| P11 | 10 Nov 2026 | A | Review of the submitted folder against the professor's class notes | verbatim, below | review: eight gaps, inconsistencies, likely questions |
| P12 | 10 Nov 2026 | A | Turn the review into a document | verbatim, below | Claude doc "Group 2 Submission Review – Smart City Traffic" |
| P13 | 12 Nov 2026 | C | Close the review's gaps | verbatim, below | the changes listed under P13 |
| P14 | 12 Nov 2026 | C | Names and push | verbatim, below | names added; pushed to GitHub |
| P15 | 12 Nov 2026 | C | Ownership by dashboard view | verbatim, below | slide 3 and the spec's team table reorganised by view |
| P16 | 12 Nov 2026 | C | Dashboard: humanoid pedestrians, single-junction focus, slider fix | verbatim, below | `sim_view.py`, `app.py`, `viz.py` |
| P17 | 12 Nov 2026 | C | Slide 16 wording and this prompt log | verbatim, below | "Our own work" column on slide 16; this log completed from Chat A |

**Chat A** — a teammate's chat (shared with the group): the first build (24 Oct), a critique and
an alternative build (7–8 Nov), and the review of the submitted folder (10 Nov).
**Chat B** — the chat that turned the 24 Oct build into the submitted build (8 Nov): dashboard
upgrade, research extensions, deck, pedestrian trade-off, value of time, corrections, report and
spec v2.0. *The exported text of this chat skips its middle part: whoever ran it should paste
P8, P9 and the first request of P10 here.*
**Chat C** — the final fixes (12 Nov).

---

## Chat A

### P1 (24 Oct, verbatim)
> There is a folder in this called group project I need you to go through it and help me build it
> first read it, understand it - there is a more content in the RDMU folder attached to get better context

### P2 (24 Oct, answers to the AI's two scoping questions)
- *Which parts do you want me to build?* → **Code + data, Dashboard, Requirement spec** (the presentation deck was left to the group at that point).
- *What should the dashboard run on?* → **Streamlit app (Recommended)**.

### P3 (24 Oct, verbatim)
> continue

During this run the AI also created the requirement specification as a Claude doc
("Requirement Specification – Smart City Traffic Management (Group 2)") without a separate prompt.

### P4 (7 Nov, verbatim)
> I want you to go through the subject outline(especially the 3.2 section) in the folder and have a glance at the class ppts
>
> I want you to assume roles - scrutinize, recommendations, ways to make better, things to add, what should be different, UI, suggest me stuff
>
> the dashboard is running on chrome at - http://127.0.0.1:8503/

### P5 (7–8 Nov, verbatim)
> Yes go ahead and fix and improve

### P11 (10 Nov, verbatim)
> I just got to know that these are what the prof said in class
>
> For now I dont want you to make any changes to the project
> I have added a new folder in that folder called smart_traffic_final this is the one we have submitted
> I want you to go through these and get back to me

### P12 (10 Nov, verbatim)
> Create a Claude Doc from your last answer.

## Chat B

### P6 (8 Nov, verbatim)
> See if there are any caveats in the dashboard, fix them, add additional functionality if possible which is around the problem statement for the project. Also make the dashboard look amazing, it should look as if it's made to attract attention, the simulators and all, the live traffic and everything it shouldn't look like a demo project. Moreover, for the simulators, and what-if if there is any correlation possible in between the sliders add them too accordingly.

### P7 (8 Nov, verbatim)
> Can we fill these 5 gaps instead of making it a weakness for our project? Also for the final dashboard after making changes, I would need PPT accordingly

### P8, P9 (≈7-8 Nov)
> **paste the verbatim prompts** for the pedestrian trade-off and value of time (P8) and the
> corrections file (P9) from Chat B.

### P10 (8 Nov)
First request (the one answered with *"I'll do both: build the 3–4 page report PDF first … then
update the specification doc"*):
> **paste the verbatim prompt**

Follow-ups (verbatim):
> so everything is in the zip now?

> Do it then keep it in the zip only

The AI could not open the Claude-doc spec from Chat B, so it wrote a new specification from the
project (v2.0) and put it in the zip with the 3-page report.

## Chat C

### P13 (12 Nov, verbatim; attached: `smart_traffic_final.zip` and `Group_2_Submission_Review___Smart_City_Traffic.pdf`)
> Read the whole thing check which gaps are needed to fill then provide the final zip with everything covered.

What P13 changed:
- **Colab link** (open to anyone, opens the notebook straight from the public GitHub repository):
  https://colab.research.google.com/github/Sbot-c/Smart_Traffic_U/blob/main/Smart_Traffic_MARL_DecisionNetwork.ipynb
- **Deck**: hosted dashboard and Colab links on the last slide (no "streamlit run"), wake-button
  note; team-and-roles slide; GenAI frames and a GenAI slide; "LIVE" → "LIVE SIMULATION";
  280 + 80 episodes (not 360); both decision-network counts explained; reference article cited;
  new uncertainty slide; Simulation Lab and pedestrian guard moved to backup; value of time and
  AHP/TOPSIS out of the main talk; fresh City Overview screenshot.
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

### P14 (12 Nov, verbatim)
> You can push it, I have linked the github repo anyways.
> Names:
> 1. Siddal 2. Shashaank 3. Sparsh 4. Aradhana 5. Shirley

What P14 changed: team names on slides 1-3 and in the spec; this folder pushed to the GitHub
repository (live app and Colab link updated).

### P15 (12 Nov, verbatim)
> We have taken responsibility from dashboard tabs itself:
> 1. City Overview - 2. RL Simulation - Shashaank 3. Decision Network Visualizer - Siddal 4. Scenario Explorer - 5. KPI & Tradeoff Analysis - Shirley 6. Policy Comparison 7. Simulation Lab 8. Research Extensions - Shashaank
> The ones empty you can fill randomly with Sparsh and Aradhana

What P15 changed: the AI assigned the empty views (City Overview and Policy Comparison to Sparsh;
Scenario Explorer and Simulation Lab to Aradhana) and rebuilt slide 3, slide 2 and the spec's team
table around the eight views.

### P16 (12 Nov, verbatim)
> Change in the dashboard:
> 1. Change the pedestrian dots into humanoid figures just like you change the cars and buses etc
> 2. Give control over single junctions as well (meaning we can choose the junction and see things for a single junction
>
> One slider is overlapping here *(with a screenshot)*

What P16 changed: pedestrian figures in the replay (waiting and crossing); a *Junction focus*
selector and click-to-zoom on the replay, with junction-level counters, KPIs, policy comparison,
approach queues and time series; the replay's control bar wraps in narrow frames.

### P17 (12 Nov, verbatim)
> In ppt, what can we write for slide 16 also what is missing in the GenAI prompt md?
>
> This conversation contains the prompts used by my teammate: *(link to Chat A)*
>
> Give me some suggestions on what to write on slide 16 for my group
>
> B
>
> This is the whole chat: *(Chat A attached as text)*
>
> Is this the one? *(Chat B attached as text)*

What P17 changed: the AI suggested three wordings for slide 16 and the group chose option B (one
line per person, based on the dashboard views each person took responsibility for); this log
was completed with the Chat A prompts (P3-P5, P11, P12) and the Chat B prompts that the exported
text shows (P6 and the P10 follow-ups).

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
| Presentation | `presentation/Smart_Traffic_Group2.pptx` | Yes — drafted by AI (framed in amber); team names and roles from the group |
| Project report | `report/Smart_Traffic_Project_Report.pdf` | Yes — drafted by AI |
| Submission review (10 Nov) | Claude doc "Group 2 Submission Review – Smart City Traffic" | Yes — drafted by AI |

## What the group should do before submitting

- Paste the remaining Chat B prompts above (P8, P9 and the first request of P10).
- Check that each person's responsibilities on slide 3 and in the spec's team table match
  what they actually did (ownership by dashboard view).
- Each person checks that their line on slide 16 ("Our own work") is true, and removes the
  amber frame from any slide the group rewrote in its own words.
- Be ready to explain every modelling choice marked **ASSUMPTION** in `config.py`.
