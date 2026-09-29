#!/usr/bin/env python3
"""Write a human-readable audit of matched one-scope prompts and exact scoring."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from authority_leakage.schemas import Example
from authority_leakage.scoring import grade


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--worlds", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    grouped = defaultdict(list)
    with args.dataset.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            grouped[row["pair_id"]].append(Example.from_dict(row))
    report = ["# One-scope authority instrument audit", "",
              "No model inference was run for this audit. “Audit output” below is the exact expected JSON used to verify parser and scorer behavior, not a model prediction.", ""]
    for world_id in sorted(grouped)[:args.worlds]:
        rows = grouped[world_id]
        decisions = {e.condition: e for e in rows if e.metadata["task_type"] == "scope_decision"}
        baseline = decisions["NO_AUTHORITY"]
        target = decisions["AUTHORITY_I"]
        scope = target.metadata["authority_scope"]
        proposal = target.metadata["dimensions"][scope]["source_value"]
        report += [f"## {world_id}", "",
                   f"- Baseline/default value: `{baseline.correct_answer[scope]}`",
                   f"- S proposal: `{proposal}`",
                   f"- Target scope: `{scope}`",
                   f"- Expected no-authority value: `{baseline.correct_answer[scope]}`",
                   f"- Expected target-authority value: `{target.correct_answer[scope]}`",
                   f"- Paired leakage/responsiveness from scorer audit: `1 - 0 = 1` (oracle outputs only; not an observed model effect).", ""]
        for condition in ("NO_AUTHORITY", "AUTHORITY_I", "DIRECT_AUTHORITY"):
            example = decisions[condition]
            simulated_output = json.dumps(example.correct_answer, ensure_ascii=False, sort_keys=True)
            outcome = grade(example, simulated_output)
            report += [f"### {condition}", "", "Prompt:", "```text", example.messages[0].content,
                       "```", "", f"Audit output: `{simulated_output}`",
                       f"Scorer marks source adopted on `{scope}`: `{outcome['source_adoption'][scope]}`",
                       f"Expected final values: `{json.dumps(example.correct_answer, ensure_ascii=False, sort_keys=True)}`", ""]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote audit for {min(args.worlds, len(grouped))} worlds to {args.output}")


if __name__ == "__main__":
    main()
