"""Paired difference A - B on identical worlds (failures count as the budget). Negative = A needs fewer steps.

  python scripts/paired_diff.py runs/eval/main search_eig search_map
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.stats import GROUPS, paired  # noqa: E402


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    run, a, b = sys.argv[1:4]
    rows = [json.loads(l) for l in open(Path(run) / "results.jsonl")]
    print(f"{a} - {b}")
    for name, tiers in GROUPS:
        d = paired(rows, a, b, tiers)
        if d:
            print(f"{name:10} n={d['n']:3}  {d['mean']:+6.1f} [{d['lo']:+6.1f}, {d['hi']:+6.1f}]  "
                  f"{d['wins']} wins / {d['losses']} losses")


if __name__ == "__main__":
    main()
