"""The leaderboard averages the seeds of one setting (split names differ only by -s<seed>) and
ignores older runs set aside next to a method's directory (runs/<method>_judge05)."""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lethe.report import load_runs, summarize  # noqa: E402


def _run(root, seed, dirname, method, score, routing=None):
    d = root / f"seed{seed}" / "runs" / dirname
    d.mkdir(parents=True)
    (d / "results.json").write_text(json.dumps({
        "track": "fact", "split": f"lethe-fact-dev-s{seed}", "model_id": "m", "n_requests": 50, "seed": seed,
        "method": {"name": method, "family": "x"}, "aggregates_tf": {"LETHE": score},
        "cost": {"unlearn_seconds": 1.0, "inference_overhead": 1.0}, **({"routing": routing} if routing else {})}))
    return d / "results.json"


def test_seeds_are_averaged_and_set_aside_runs_ignored(tmp_path):
    paths = [_run(tmp_path, s, "memit", "memit", 10.0 * s) for s in (1, 2, 3)]
    paths.append(_run(tmp_path, 1, "memit_judge05", "memit", 99.0))
    paths.append(_run(tmp_path, 1, "noop.incomplete_20261024", "noop", 5.0))
    paths.append(_run(tmp_path, 1, "memit_x", "memit_x", 7.0))          # its own method, not an aside copy of memit
    runs = load_runs(paths)
    assert sorted(Path(r["_path"]).parent.name for r in runs) == ["memit", "memit", "memit", "memit_x"]
    table = summarize(runs)
    row = next(v for v in table.values() if v["method"] == "memit")
    assert row["seeds"] == [1, 2, 3] and row["LETHE"][0] == 20.0
    assert "lethe-fact-dev ·" in row["track"]      # split shown under the benchmark name


def test_routing_diagnostics_average_over_seeds(tmp_path):
    """The detect-then-edit table used to print the first seed's routing numbers only."""
    def routing(recall):
        return {"by_family": {}, "forget_route_recall": recall, "forget_residual_from_routing": 100 - recall,
                "forget_residual_from_actuation": 0.0, "retain_false_fire_rate": 5.0}
    paths = [_run(tmp_path, s, "router", "router", 20.0, routing(r)) for s, r in ((1, 80.0), (2, 90.0), (3, 100.0))]
    paths.append(_run(tmp_path, 1, "ga", "ga", 5.0))
    table = summarize(load_runs(paths))
    row = next(v for v in table.values() if v["method"] == "router")
    assert row["routing"]["forget_route_recall"] == 90.0 and row["routing"]["forget_residual_from_routing"] == 10.0
    assert row["routing"]["retain_false_fire_rate"] == 5.0 and "by_family" not in row["routing"]
    assert next(v for v in table.values() if v["method"] == "ga")["routing"] is None


def test_exclude_drops_a_method(tmp_path):
    paths = [_run(tmp_path, 1, "memit", "memit", 30.0), _run(tmp_path, 1, "noop", "noop", 12.0), _run(tmp_path, 1, "ga", "ga", 5.0)]
    names = sorted(v["method"] for v in summarize(load_runs(paths)).values())
    assert names == ["ga", "memit", "noop"]
    names = sorted(v["method"] for v in summarize(load_runs(paths, exclude=["noop"])).values())
    assert names == ["ga", "memit"]
