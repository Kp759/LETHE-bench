# LETHE-Bench on one AWS GPU (g5.2xlarge)

`run_lethe_aws.sh` runs the same chain as the Wulver jobs (`scripts/slurm/`) on one machine, one step after another, without SLURM. It was written for a **g5.2xlarge**: 1× NVIDIA A10G (24 GB), 8 vCPUs, 32 GB RAM. Every step skips finished work, so after a stop, a crash or a reboot you rerun the same command.

## Commands

```bash
cd ~/workspace/Unlearning/semantic-unlearning/LETHE-bench
bash scripts/aws/run_lethe_aws.sh setup                     # once, about 30-60 min
nohup bash scripts/aws/run_lethe_aws.sh core > ~/lethe_core.log 2>&1 &
tail -f ~/lethe_core.log                                    # Ctrl-C stops watching, not the run
bash scripts/aws/run_lethe_aws.sh status                    # what is done / missing / failed
```

| Stage | Runs |
|---|---|
| `setup` | Python 3.11 environment in `~/envs/lethe` (torch 2.5.1 CUDA 12.4, transformers 4.57.1: the tested versions; `/opt/pytorch` is not touched); MQuAKE, zsRE and RWKU downloads; the 100,020-article Wikipedia corpus the Wulver covariance uses; the test suite |
| `core` | ROME / MEMIT / ZeroUnlearn covariance (once); per dataset: build the split, then 13 methods (6 reference cheats, GA, GD, NPO, RMU, ROME, MEMIT, ZeroUnlearn); red-team gate; leaderboards |
| `agent` | Agentic leakage track (`AGENT_TRACK.md`) for 13 methods |
| `sequential` | Sequential deletion: GD, NPO, MEMIT over 5 batches |
| `all` | `core`, `agent`, `sequential` |

## Datasets

`DATASETS` (default `merged mcf mquake zsre`) builds one split per entry:
- **merged:** facts from MultiCounterFact, zsRE and MQuAKE together. This is LETHE-Bench's headline split and the one the Wulver runs use.
- **mcf / mquake / zsre:** one dataset each, 50 requests from that dataset alone.

A per-dataset breakdown of the merged split would not work: its 50 requests come out about 32 MQuAKE, 17 MultiCounterFact and 1 zsRE.

zsRE facts get fewer probe types. zsRE has no relation templates, so there are no cross-lingual, multi-hop or same-subject probes, and its split is the easiest to game. The red-team gate is reported per split, so that shows up in the results.

RWKU supplies the utility questions in every split. Its entity track (forget a whole person) is not run here: ROME, MEMIT and ZeroUnlearn take fact requests only.

## Changes for the 24 GB GPU

These are recorded in every `results.json` (method arguments, `eval_config`):
- **GA, GD and NPO** train the MLP down-projections in mixed precision: a bf16 forward pass and fp32 copies of the trained weights. On Wulver they train every MLP weight with the whole model in fp32. That needs about 38 GB for the 3B model; down-projections alone in fp32 still need about 21 GB before activations.
- **The relearning attack** fine-tunes the down-projections (Wulver: every MLP weight in bf16, about 23 GB).

Everything else is identical to the Wulver runs, including the covariance corpus, the edit hyperparameters and the evaluation.

## Rough time (A10G, Llama-3.2-3B, seed 1)

These are estimates. The A10G is about 2-3× slower than the A100s the Wulver numbers come from; expect ±50%.

| Step | Time |
|---|---|
| `setup` | 30-60 min |
| Covariance, once | 3-5 h |
| `core`, per dataset | 13-19 h (build ~1 h, 13 methods ~8-10 h) |
| `agent`, merged | 10-15 h |
| `sequential`, merged | 5-8 h |

A g5.2xlarge costs about $1.2 per hour on demand. Start with the headline split, check it, then run the rest:

```bash
DATASETS=merged nohup bash scripts/aws/run_lethe_aws.sh core > ~/lethe_core.log 2>&1 &
# later: the per-dataset splits, then the other stages
DATASETS="mcf mquake zsre" nohup bash scripts/aws/run_lethe_aws.sh core > ~/lethe_core2.log 2>&1 &
```

## Outputs

```
results/aws/<dataset>/llama32_3b/dev/seed1/
  split.json  split_summary.md  validation.json
  runs/<method>/results.json, probes.jsonl      runs/RED_TEAM_REPORT.md
  agent/<method>/agent.json, episodes.jsonl     agent/agent_leaderboard.md
results/aws/<dataset>/llama32_3b/dev/leaderboard/leaderboard.md
results/aws/leaderboard_llama32_3b/leaderboard.md       every dataset, with a method x dataset LETHE-Bench matrix
results/aws/logs/                                       one log per step (not committed)
```

To share results, commit them; caches and logs are ignored:

```bash
git add results/aws && git commit -m "LETHE-Bench AWS results" && git push
```

## Settings

| Variable | Default |
|---|---|
| `MODEL_PATH` | `~/models/Llama-3.2-3B-Instruct` |
| `DATASETS` | `merged mcf mquake zsre` |
| `AGENT_DATASETS` / `SEQ_DATASETS` | `merged` |
| `SEEDS` | `1` |
| `N_FORGET` | `50` |
| `METHODS` / `AGENT_METHODS` / `SEQ_METHODS` | see the script header |
| `METHOD_SECONDS` | `14400` |
| `MOM2_TEXT` | `~/datasets/wikipedia_20231101_100k` |
| `ENV_DIR` | `~/envs/lethe` |
| `DRY_RUN=1` | print the commands instead of running them |
