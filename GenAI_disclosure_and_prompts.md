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

The project was built in eight Claude chats (A-H, described below the table). Prompts marked
**verbatim** are copied exactly from the chat. Two are still missing because the exported text
of those chats does not include them: **P6** and the first request of **P17**.

| # | Date | Chat | Purpose | Prompt | Output |
|---|---|---|---|---|---|
| P1 | 24 Oct 2026 | A | Build the project | verbatim, below | simulator, agents, decision network, experiments, data, notebook, first dashboard |
| P2 | 24 Oct 2026 | A | Answers to the AI's two scoping questions | verbatim, below | scope: code + data, dashboard, spec; Streamlit |
| P3 | 24 Oct 2026 | A | Continue | verbatim, below | rest of the 24 Oct build, and the requirement specification drafted as a Claude doc (no separate prompt) |
| P4 | 7 Nov 2026 | A | Critique against the subject outline (section 3.2) and class decks | verbatim, below | review from several roles: fair comparison, model selection, course techniques, UI |
| P5 | 7–8 Nov 2026 | A | Apply the critique | verbatim, below | an alternative 8 Nov build and a revised Claude-doc spec (**not** the submitted build) |
| P6 | ≈7 Nov 2026 | D | Fix the three priority items from the critique | **paste the verbatim prompt** | validation-based model selection; fair signal-vs-routing comparison in the notebook, README and Policy Comparison; `course_methods.py` (uncertainty criteria, risk-averse utility, MDP table, price of anarchy, AHP, TOPSIS); `CORRECTIONS_FOR_SPEC_AND_SLIDES.md` |
| P7 | ≈7 Nov 2026 | D | Acknowledge | verbatim, below | AI suggested next steps |
| P8 | ≈7 Nov 2026 | D | Build the live what-if panel | verbatim, below | *Live What-If* tab (later replaced by the Simulation Lab) |
| P9 | ≈7 Nov 2026 | D | Ask for further improvements | verbatim, below | ranked to-do list (UI fixes, side-by-side replay, pedestrian trade-off, value of time, extras) |
| P10 | ≈7–8 Nov 2026 | F | Items 1 and 2 of the Chat D list (pasted in), with the problem statement attached | verbatim, below | only the open view drawn, projector fonts, Deploy menu hidden, readable tables, tie label, readings under value of information and the weight triangle; side-by-side replay with live counters, queue-coloured roads, minute slider |
| P11 | ≈7–8 Nov 2026 | G | Items 1-4 of the Chat D list (pasted in), with the problem statement attached | verbatim, below | an attempt at all four items (UI fixes, replay, pedestrian trade-off, value of time); the trade-off and value of time that were shipped came from Chat H |
| P12 | ≈8 Nov 2026 | H | Items 3 and 4 of the Chat D list (pasted in), with the problem statement attached | verbatim, below | `value_of_time.py` (AED 51.8 per person-hour, U.S. DOT value × AED peg); `ped_tradeoff.py` (seven pedestrian weights × 3 training seeds) |
| P13 | ≈8 Nov 2026 | H | Ship what is finished | verbatim, below | `smart_traffic_updated.zip` with `ped_tradeoff.py`, `value_of_time.py`, `make_figures.py`, `data/ped_weight_tradeoff.csv` (Q-learning sweep; the half-finished SARSA sweep stopped), `figures/`, two new KPI-view sections |
| P14 | ≈8 Nov 2026 | E | Finish what was left of the Chat H work (Chat H's status message pasted in) | verbatim, below | dashboard sections tested, trade-off figure labels fixed, paste-ready wording added to `CORRECTIONS_FOR_SPEC_AND_SLIDES.md`, README and caption updated; SARSA sweep and notebook left undone on purpose |
| P15 | 8 Nov 2026 | B | Dashboard upgrade | verbatim, below | command-centre theme, Simulation Lab, linked sliders |
| P16 | 8 Nov 2026 | B | Research extensions and the deck | verbatim, below | `extensions.py`, `data/ext_*.csv`, Research Extensions view, 16-slide deck |
| P17 | 8 Nov 2026 | B | Project report and spec v2.0 PDF | first request: **paste the verbatim prompt**; two follow-ups verbatim, below | `report/Smart_Traffic_Project_Report.pdf`, `Requirement_Specification.pdf` v2.0 |
| P18 | 10 Nov 2026 | A | Review of the submitted folder against the professor's class notes | verbatim, below | review: eight gaps, inconsistencies, likely questions |
| P19 | 10 Nov 2026 | A | Turn the review into a document | verbatim, below | Claude doc "Group 2 Submission Review – Smart City Traffic" |
| P20 | 12 Nov 2026 | C | Close the review's gaps | verbatim, below | the changes listed under P20 |
| P21 | 12 Nov 2026 | C | Names and push | verbatim, below | names added; pushed to GitHub |
| P22 | 12 Nov 2026 | C | Ownership by dashboard view | verbatim, below | slide 3 and the spec's team table reorganised by view |
| P23 | 12 Nov 2026 | C | Dashboard: humanoid pedestrians, single-junction focus, slider fix | verbatim, below | `sim_view.py`, `app.py`, `viz.py` |
| P24 | 12 Nov 2026 | C | Slide 16 wording and this prompt log | verbatim, below | "Our own work" column on slide 16; this log completed |

Listed in the order the work happened:

- **Chat A** — a teammate's chat (shared with the group): the first build (24 Oct), a critique
  and an alternative build (7–8 Nov), and the review of the submitted folder (10 Nov).
- **Chat D** — took the 24 Oct build and the critique's priority list (≈7 Nov): fair comparison,
  model selection, course methods, corrections file, live what-if panel, and the ranked to-do
  list used in Chats F and G. *Its first request (P6) is above the exported text: scroll to the
  top of the chat (the AI's reply begins "Read the code and confirmed your numbers against the
  CSVs…").*
- **Chats F, G and H** — three fresh chats, each started by pasting the Chat D to-do list: F
  asked for items 1-2 (UI fixes, side-by-side replay), G for items 1-4, H for items 3-4
  (pedestrian trade-off, value of time) (≈7–8 Nov). The shipped trade-off and value-of-time
  files come from H.
- **Chat E** — a short follow-up that finished the Chat H work (≈8 Nov), started from a pasted
  copy of Chat H's last status message.
- **Chat B** — turned that into the submitted build (8 Nov): dashboard upgrade, research
  extensions, deck, report and spec v2.0. *The exported text skips its middle: the first report
  request (P17) is the message just before the reply "I'll do both: build the 3–4 page report
  PDF first…".*
- **Chat C** — the final fixes (12 Nov).

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

### P18 (10 Nov, verbatim)
> I just got to know that these are what the prof said in class
>
> For now I dont want you to make any changes to the project
> I have added a new folder in that folder called smart_traffic_final this is the one we have submitted
> I want you to go through these and get back to me

### P19 (10 Nov, verbatim)
> Create a Claude Doc from your last answer.

## Chat D

### P6 (≈7 Nov)
> **paste the verbatim prompt** (the first message of Chat D, which passed on the critique's
> three priority fixes: compare packages fairly, choose tables on validation runs, cover more
> course sessions)

### P7 (≈7 Nov, verbatim)
> Ok

### P8 (≈7 Nov, verbatim)
> Sure you can work on it

(The AI had offered to build the live what-if panel.)

### P9 (≈7 Nov, verbatim)
> what else can we improvise in this

## Chat F

### P10 (≈7–8 Nov, verbatim; attached: the problem statement PDF)
> Do these changes namely(1,2) , I have also attached the problem statement pdf for context

Items 1 and 2 of the AI's list in reply to P9, pasted into this new chat below the request: *1. UI fixes* (draw only the open tab, bigger
fonts, hide the Deploy menu, round table numbers and readable column names, fix the "best" label
on a weighted-sum tie, one-line readings under the value-of-information chart and the weight
triangle) and *2. Side-by-side replay* (fit to the screen, live counters for queued cars, delay
and CO₂, roads coloured by queue length, replay clock linked to the minute slider).

## Chat G

### P11 (≈7–8 Nov, verbatim; attached: the problem statement PDF)
> Do these changes namely(1,2,3 and 4) , I have also attached the problem statement pdf for context

followed by the same pasted list: items 1-2 as in P10, *3. Pedestrian-weight trade-off chart*
(retrain rush hour at several pedestrian weights, plot delay against pedestrian wait) and
*4. Value of time in money* (convert vehicle-hours saved into currency with a cited value of
time).

## Chat H

### P12 (≈8 Nov, verbatim; attached: the problem statement PDF)
> Do these changes namely(3 and 4) , I have also attached the problem statement pdf for context

followed by the pasted list from item 3 onwards (*3. Pedestrian-weight trade-off chart*, *4. Value
of time in money*, then the cheap extras and the demo items).

### P13 (≈8 Nov, verbatim)
> Just give me the changed zip whatever is left I'll do it later

The AI stopped the half-finished SARSA sweep, shipped the complete Q-learning sweep and listed
what was left (the corrections wording, the SARSA sweep, the notebook). That status message is
the text pasted into Chat E.

## Chat E

### P14 (≈8 Nov, verbatim)
> Whatever is left in the below texts, do those changes

followed by a pasted copy of Chat H's last status message (the Q-learning pedestrian-weight
sweep shipped without the SARSA half-curve; new `ped_tradeoff.py`, `value_of_time.py`,
`make_figures.py`, `data/ped_weight_tradeoff.csv`, `figures/`; items left: the wording in
`CORRECTIONS_FOR_SPEC_AND_SLIDES.md`, the SARSA sweep, the notebook; caveats on the US value of
time and the 64.8 vs 69 s retraining gap).

## Chat B

### P15 (8 Nov, verbatim)
> See if there are any caveats in the dashboard, fix them, add additional functionality if possible which is around the problem statement for the project. Also make the dashboard look amazing, it should look as if it's made to attract attention, the simulators and all, the live traffic and everything it shouldn't look like a demo project. Moreover, for the simulators, and what-if if there is any correlation possible in between the sliders add them too accordingly.

### P16 (8 Nov, verbatim)
> Can we fill these 5 gaps instead of making it a weakness for our project? Also for the final dashboard after making changes, I would need PPT accordingly

### P17 (8 Nov)
First request (the one answered with *"I'll do both: build the 3–4 page report PDF first … then
update the specification doc"*):
> **paste the verbatim prompt**

Follow-ups (verbatim):
> so everything is in the zip now?

> Do it then keep it in the zip only

The AI could not open the Claude-doc spec from Chat B, so it wrote a new specification from the
project (v2.0) and put it in the zip with the 3-page report.

## Chat C

### P20 (12 Nov, verbatim; attached: `smart_traffic_final.zip` and `Group_2_Submission_Review___Smart_City_Traffic.pdf`)
> Read the whole thing check which gaps are needed to fill then provide the final zip with everything covered.

What P20 changed:
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

### P21 (12 Nov, verbatim)
> You can push it, I have linked the github repo anyways.
> Names:
> 1. Siddal 2. Shashaank 3. Sparsh 4. Aradhana 5. Shirley

What P21 changed: team names on slides 1-3 and in the spec; this folder pushed to the GitHub
repository (live app and Colab link updated).

### P22 (12 Nov, verbatim)
> We have taken responsibility from dashboard tabs itself:
> 1. City Overview - 2. RL Simulation - Shashaank 3. Decision Network Visualizer - Siddal 4. Scenario Explorer - 5. KPI & Tradeoff Analysis - Shirley 6. Policy Comparison 7. Simulation Lab 8. Research Extensions - Shashaank
> The ones empty you can fill randomly with Sparsh and Aradhana

What P22 changed: the AI assigned the empty views (City Overview and Policy Comparison to Sparsh;
Scenario Explorer and Simulation Lab to Aradhana) and rebuilt slide 3, slide 2 and the spec's team
table around the eight views.

### P23 (12 Nov, verbatim)
> Change in the dashboard:
> 1. Change the pedestrian dots into humanoid figures just like you change the cars and buses etc
> 2. Give control over single junctions as well (meaning we can choose the junction and see things for a single junction
>
> One slider is overlapping here *(with a screenshot)*

What P23 changed: pedestrian figures in the replay (waiting and crossing); a *Junction focus*
selector and click-to-zoom on the replay, with junction-level counters, KPIs, policy comparison,
approach queues and time series; the replay's control bar wraps in narrow frames.

### P24 (12 Nov, verbatim)
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
>
> This: *(Chat F, P10)*
>
> This? *(the earlier part of Chat D, P6-P9)*
>
> This? *(Chat E, P14)*
>
> This? *(Chat G, P11)*
>
> This? *(Chat H, P12-P13)*

What P24 changed: the AI suggested three wordings for slide 16 and the group chose option B (one
line per person, based on the dashboard views each person took responsibility for); this log
was completed from the exported text of Chats A, B, D, E, F, G and H (P3-P21).

---

## What the AI produced

| Deliverable | Files | Generated by AI? |
|---|---|---|
| Simulator, agents (Q-learning / SARSA), baselines, decision network, experiment pipeline | `config.py`, `traffic_env.py`, `agents.py`, `decision_network.py`, `experiments.py`, `run_experiments.py`, `robustness.py`, `refresh_analysis.py`, `course_methods.py` | Yes — all code |
| Research extensions and sensing study | `extensions.py`, `sensing.py`, `ped_tradeoff.py`, `value_of_time.py`, `data/ext_*.csv` | Yes — code and the generated data |
| Course decision methods | `course_methods.py` (uncertainty criteria, risk-averse utility, MDP table, price of anarchy, AHP, TOPSIS) | Yes — code |
| Wording corrections | `CORRECTIONS_FOR_SPEC_AND_SLIDES.md` | Yes — drafted by AI |
| Colab notebook | `Smart_Traffic_MARL_DecisionNetwork.ipynb` (built by `build_notebook.py`, updated 12 Nov) | Yes — code and explanatory text |
| Data CSV files | `data/*.csv` | Generated by running the AI-written simulator |
| Streamlit dashboard | `app.py`, `viz.py`, `sim_view.py`, `lab.py`, `.streamlit/config.toml` | Yes — all code |
| Requirement specification v2.1 and UI wireframe | `Requirement_Specification.pdf`, `docs/spec_source/`, `docs/ui_wireframe.html`, `docs/dashboard_ui_diagram.png` | Yes — drafted by AI (framed in amber) |
| Presentation | `presentation/Smart_Traffic_Group2.pptx` | Yes — drafted by AI (framed in amber); team names and roles from the group |
| Project report | `report/Smart_Traffic_Project_Report.pdf` | Yes — drafted by AI |
| Submission review (10 Nov) | Claude doc "Group 2 Submission Review – Smart City Traffic" | Yes — drafted by AI |

## What the group should do before submitting

- Paste the two missing prompts above (P6 and the first request of P17).
- Check that each person's responsibilities on slide 3 and in the spec's team table match
  what they actually did (ownership by dashboard view).
- Each person checks that their line on slide 16 ("Our own work") is true, and removes the
  amber frame from any slide the group rewrote in its own words.
- Be ready to explain every modelling choice marked **ASSUMPTION** in `config.py`.
