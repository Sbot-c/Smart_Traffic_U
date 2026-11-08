"""Re-run everything that follows training, from the saved Q-tables (no retraining):

  1. choose general vs scenario-adapted tables on VALIDATION episodes  -> model_selection.csv
  2. evaluate the four policies on the held-out TEST episodes           -> kpi_results.csv
  3. component studies, representative runs, decision-network samples

Test episodes are never used to choose between policies.
"""
import gzip
import json
import time

import experiments as X

if __name__ == "__main__":
    t0 = time.time()
    models = X.load_models("data/q_tables.npz")
    sel = X.select_tables(models, X.FULL)
    sel.round(3).to_csv("data/model_selection.csv", index=False)
    kpi = X.evaluate_policies(models, X.FULL)
    kpi.round(4).to_csv("data/kpi_results.csv", index=False)
    abl = X.ablations(models, X.FULL, verbose=True)
    abl.round(4).to_csv("data/ablation_results.csv", index=False)
    sim_log, anims, decisions = X.record_runs(models)
    sim_log.to_csv("data/smart_traffic_simulation_log.csv", index=False)
    decisions.to_csv("data/agent_decisions.csv", index=False)
    with gzip.open("data/animations.json.gz", "wt") as f:
        json.dump(anims, f, separators=(",", ":"))
    dn = X.dn_samples(models, X.FULL, verbose=True)
    dn.round(4).to_csv("data/decision_network_samples.csv", index=False)
    print(f"done in {time.time() - t0:.0f} s")
