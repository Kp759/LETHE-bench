# Results

| Folder | What |
|---|---|
| `cluster/leaderboard_dev/` | Cross-model table: Llama-3.2-3B-Instruct, Qwen3-4B and Llama-3.1-8B-Instruct, dev partition, seeds 1–5, 50 requests per seed. 10 methods (GA, GD, NPO, SimNPO, trained refusal, task-vector negation, RMU, ROME, MEMIT, ZeroUnlearn) and 6 reference cheats; `leaderboard.md`, `leaderboard.json` (teacher-forced scorer), `leaderboard_gen.json` (generation scorer), `pareto.png` |
| `cluster/<model>/dev/leaderboard/` | The same per model, with the per-axis, per-family, routing, attack and cost tables |

The per-seed run directories (`results.json`, `probes.jsonl`, `RED_TEAM_REPORT.md`) stay on the cluster; `EVIDENCE.md` §3 reads the tables. They were produced by `bash scripts/submit_lethe_compare.sh full all` (`scripts/README_CLUSTER.md`) and are re-rendered from the saved tables with `python scripts/render_leaderboard.py <dir>`.

`python -m lethe toy --out results/toy` runs the whole pipeline on a synthetic world on a CPU in about ten minutes (a pipeline check, not a result; its output is not committed).
