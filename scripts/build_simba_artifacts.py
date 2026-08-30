#!/usr/bin/env python3
"""Rebuild committed DSPy artifacts from immutable templates and demos."""

import argparse
import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "scripts" / "simba_synthetic_demos.json"
FRAGMENTS = ROOT / "scripts" / "simba_signature_fragments.json"
GROUNDING_CONTRACT = ROOT / "scripts" / "simba_grounding_contract.txt"
CANONICAL_TEMPLATE = ROOT / "scripts" / "simba_canonical_template.json"
BASELINE_TEMPLATE = ROOT / "scripts" / "simba_baseline_template.json"
CANONICAL = Path("data/simba_canonical_pln.json")
BASELINE = Path("data/simba_all.json")
ROOT_BASELINE = Path("simba_all.json")
DEMO_VARIANTS = ("neutral", "final-removed")
PROMPT_VARIANTS = ("current", "hardened")


def expand_demo(source_demo: dict, pln_spec: str) -> dict:
    demo = copy.deepcopy(source_demo)
    return {
        "augmented": demo["augmented"],
        "sentences": demo["sentences"],
        "context": demo["context"],
        "pln_spec": pln_spec,
        "reasoning": demo["reasoning"],
        "statements": demo["statements"],
        "queries": demo["queries"],
    }


def load_demo_sets(variant: str = "neutral") -> tuple[list[dict], list[dict]]:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    safe_demos = [
        expand_demo(demo, fixture["pln_spec"]) for demo in fixture["safe_demos"]
    ]
    if variant == "final-removed":
        return safe_demos, safe_demos[:6]

    neutral_demo = expand_demo(fixture["neutral_demo"], fixture["pln_spec"])
    return [*safe_demos, neutral_demo], [*safe_demos[:6], neutral_demo]


def load_template(path: Path, prompt_variant: str = "current") -> dict:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    fragments = json.loads(FRAGMENTS.read_text(encoding="utf-8"))
    signature = artifact["nl2pln.predict"]["signature"]
    signature["instructions"] = "".join(
        fragments[name] for name in signature.pop("instruction_fragments")
    )
    if prompt_variant == "hardened":
        signature["instructions"] += GROUNDING_CONTRACT.read_text(encoding="utf-8")
    # DSPy serializes instructions before fields.
    artifact["nl2pln.predict"]["signature"] = {
        "instructions": signature["instructions"],
        "fields": signature["fields"],
    }
    return artifact


def build(template: Path, demos: list[dict], prompt_variant: str = "current") -> bytes:
    artifact = load_template(template, prompt_variant)
    predictor = artifact["nl2pln.predict"]
    artifact["nl2pln.predict"] = {
        "traces": predictor["traces"],
        "train": predictor["train"],
        "demos": demos,
        "signature": predictor["signature"],
        "lm": predictor["lm"],
    }
    return (json.dumps(artifact, indent=2, ensure_ascii=False) + "\n").encode()


def expected_artifacts(
    demo_variant: str = "neutral", prompt_variant: str = "current"
) -> dict[Path, bytes]:
    canonical_demos, baseline_demos = load_demo_sets(demo_variant)
    baseline = build(BASELINE_TEMPLATE, baseline_demos, prompt_variant)
    return {
        CANONICAL: build(CANONICAL_TEMPLATE, canonical_demos, prompt_variant),
        BASELINE: baseline,
        ROOT_BASELINE: baseline,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true", help="fail if committed artifacts need rebuilding"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="write under another directory instead of the repository root",
    )
    parser.add_argument(
        "--demo-variant",
        choices=DEMO_VARIANTS,
        default="neutral",
        help="select the demonstration set (default: neutral)",
    )
    parser.add_argument(
        "--prompt-variant",
        choices=PROMPT_VARIANTS,
        default="current",
        help="select the signature instructions (default: current)",
    )
    args = parser.parse_args()

    artifacts = expected_artifacts(args.demo_variant, args.prompt_variant)
    output_root = args.output_dir or ROOT
    if args.check:
        stale = []
        for path, data in artifacts.items():
            target = output_root / path
            if not target.is_file() or target.read_bytes() != data:
                stale.append(str(path))
        if stale:
            parser.error("missing or stale artifacts: " + ", ".join(stale))
        return 0

    for path, data in artifacts.items():
        destination = output_root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
