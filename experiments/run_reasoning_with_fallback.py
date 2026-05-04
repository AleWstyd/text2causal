"""Merge Reactome+LLM priors with free-text claims for the no-context stratum."""

from __future__ import annotations

import json
import os
from pathlib import Path

from reasoning.merge_freetext_fallback import (
    infer_column_set_from_priors,
    merge_with_freetext_fallback,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

JOBS: tuple[tuple[str, Path, Path, Path], ...] = (
    (
        "sachs",
        REPO_ROOT / "experiments" / "causal_priors_sachs.json",
        REPO_ROOT / "experiments" / "freetext_priors_sachs.json",
        REPO_ROOT / "experiments" / "causal_priors_sachs_with_fallback.json",
    ),
    (
        "dream4_psn",
        REPO_ROOT / "experiments" / "causal_priors_dream4_psn.json",
        REPO_ROOT / "experiments" / "freetext_priors_dream4_psn.json",
        REPO_ROOT / "experiments" / "causal_priors_dream4_psn_with_fallback.json",
    ),
    (
        "liverdream",
        REPO_ROOT / "experiments" / "causal_priors_liverdream.json",
        REPO_ROOT / "experiments" / "freetext_priors_liverdream.json",
        REPO_ROOT / "experiments" / "causal_priors_liverdream_with_fallback.json",
    ),
)


def _atomic_write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    for name, causal_path, ft_path, out_path in JOBS:
        if not causal_path.is_file():
            print(f"[skip {name}] missing {causal_path}")
            continue
        if not ft_path.is_file():
            print(f"[skip {name}] missing {ft_path} (no free-text priors)")
            continue
        causal = json.loads(causal_path.read_text(encoding="utf-8"))
        if not isinstance(causal, dict):
            raise TypeError(f"Expected object in {causal_path}")
        freetext = json.loads(ft_path.read_text(encoding="utf-8"))
        if not isinstance(freetext, dict):
            raise TypeError(f"Expected object in {ft_path}")
        col = infer_column_set_from_priors(causal)
        merged = merge_with_freetext_fallback(causal, freetext, column_set=col)
        _atomic_write_json(out_path, merged)
        n_fb = merged.get("n_fallback_applied", 0)
        print(f"[{name}] wrote {out_path} (n_fallback_applied={n_fb})")


if __name__ == "__main__":
    main()
