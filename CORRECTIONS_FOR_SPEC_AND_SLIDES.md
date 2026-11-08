# Wording to use in the requirement specification and the slides

The specification is a Claude doc that I could not open, so these are replacement sentences to paste in.

**Replace** "Q-learning beats actuated by 21 % in rush hour" **with:**
> Compared as complete packages, the Q-learning system (RL signals + cooperative AV routing) has 26 % lower delay than the actuated package (actuated signals + static routes) in rush hour. Most of that gap comes from routing: with the same cooperative routing, Q-learning signals (64.8 s) and actuated signals (66.9 s) are statistically indistinguishable in the daily peak (paired p = 0.08). In the event surge, RL signals win clearly (46.6 vs 62.9 s, −26 %).

**One-line story for the slides:** Cooperative routing is the big lever. RL signals pay off when demand is unusual. Actuated signals are hard to beat in the daily peak. The decision network recommends exactly that.

**Replace** any claim that RL learning curves improve in rush hour **with:**
> During rush-hour adaptation the training delay rose (about 72 to 82 s). Extra rush-hour training made the policy worse, so the general table (pre-trained on all scenarios) is deployed in rush hour. The choice between general and adapted tables is made on separate validation episodes, never on the test episodes (Q-learning 66.2 vs 68.9 s on validation; 64.8 vs 69.1 s on test).

**Q-learning vs SARSA caption:** the gap between the algorithms comes mostly from training randomness; re-adapting with three other seeds put SARSA at 68–70 s and Q-learning at 71–75 s in rush hour.

**Limitations to state:** 360 training episodes for RL against an untuned actuated rule; "safety" means expected conflicts, not crashes; AV speed control and lane selection are not modelled (routing and platooning only); 25 % AV share; cooperative routing needs vehicle-to-infrastructure communication.

**Why lookup tables, not deep RL (one slide):** 2,250 states per junction, every decision can be inspected in the Q-table, training takes minutes on a CPU. Deep RL would fit if the state grew to raw detector or camera data.

**Q&A answers verified in this project:** retiming the fixed plan for the peak does not help it (109.5 s vs 109.3 s). The RL signals almost never fall back to the actuated rule (at most 7 of 25,920 decisions per scenario).

---

## Added: pedestrian trade-off and value of time (wording to paste)

**Pedestrian trade-off (slide + spec).** Use this sentence:
> We treat the pedestrian weight in the signal reward as a design choice, not a constant. Retraining the Q-learning system at weights from 0 to 8 (3 training runs each, same 8 test episodes), pedestrian wait falls only from 23.9 s to 22.2 s (about 7 %) while vehicle delay rises from 68.9 s to 79.3 s (about 15 %). Between weights 0 and 0.5 nothing measurable changes (about 69 s delay, 24 s pedestrian wait), so the default of 0.5 is a modest setting, not a protective one.

**What the figure shows (say this if asked):**
* With the same cooperative routing, actuated signals (66.9 s vehicle delay, 21.2 s pedestrian wait) are better on **both** axes than every Q-learning weight. Fixed-time signals give the shortest pedestrian wait (19.0 s) but the worst vehicle delay (92.1 s).
* The curve is not perfectly smooth (weight 4 is slightly better for pedestrians than weight 8, and jaywalks do not fall steadily). The spread across training runs is as large as many gaps between neighbouring weights, so quote the overall slope, not single points.
* In money, no weight beats actuated signals: net benefit is about -AED 330 per peak window at weight 0 and about -AED 370 at weight 0.5 (spread across runs about 300), falling to about -AED 1,020 at weight 8. Extra pedestrian protection is paid for in vehicle time. Whether it is worth it is a city policy decision (it depends on the weight put on pedestrians), which is exactly what the decision network is for.

**Value of time (slide + spec).** Use this sentence:
> Delay is converted to money with the U.S. DOT value of travel time (USD 14.10 per person-hour, 2015 dollars, converted at the AED peg of 3.6725, which gives AED 51.8), counting driver-only vehicles and valuing pedestrian waiting time at twice the in-vehicle rate. In the daily peak, the Q-learning package saves about 66 vehicle-hours per peak window against fixed-time signals with static routes, worth about AED 3,440; the actuated package saves about 33 vehicle-hours (AED 1,690).

**Say these limits out loud:**
* The AED 51.8 per person-hour is a **US value**, not a Dubai one. A Dubai income-based value would probably be higher, but the figure is also not adjusted for inflation since 2015.
* One person per vehicle is a conservative choice; buses and passengers are not counted.
* Any yearly figure depends on the assumption of 500 peak windows per year (two peaks x 250 working days). It is a parameter in the dashboard, not a measurement.
* Savings are against fixed-time signals with static routes. Against actuated signals with the same routing, Q-learning's gain in the daily peak is small (AED 3,440 vs 3,280 per window) and not statistically significant (paired p = 0.08).

**Retraining caveat (Q&A).** Retraining at the default pedestrian weight 0.5 gives about 69 s rush-hour delay (spread across three runs about 4 s), not the headline 64.8 s. The headline is the deployed model from the main pipeline. The gap is training randomness from retraining from scratch, so say so if asked. The three retraining runs at weight 0.5 gave 64.8 s (training seed 0, the same as the headline), 72.4 s and 70.1 s, so the headline is the best of three. That is another reason not to present the rush-hour Q-learning vs actuated result (64.8 vs 66.9 s) as a win.

**Not done in this version:** the SARSA sweep (an unfinished sweep would show a misleading half-curve, so only the complete Q-learning sweep ships) and a rebuilt Colab notebook (rebuilding would erase its saved outputs).
