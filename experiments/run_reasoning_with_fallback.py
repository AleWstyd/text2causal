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

        causal = json.loads(causal_path.read_text(encoding="utf-8"))
        if not isinstance(causal, dict):
            raise TypeError(f"Expected object in {causal_path}")

        per_pair_path = (
            REPO_ROOT / "experiments" / f"per_pair_freetext_priors_{name}.json"
        )

        has_para = ft_path.is_file()
        has_pp = per_pair_path.is_file()

        if not has_para and not has_pp:
            print(
                f"[skip {name}] missing both {ft_path.name} and "
                f"{per_pair_path.name} (no free-text priors)"
            )
            continue

        col = infer_column_set_from_priors(causal)

        current = causal
        fb_total: list[list[str]] = []
        n_para = 0
        n_pp = 0

        if has_para:
            freetext = json.loads(ft_path.read_text(encoding="utf-8"))
            if not isinstance(freetext, dict):
                raise TypeError(f"Expected object in {ft_path}")
            current = merge_with_freetext_fallback(
                causal,
                freetext,
                column_set=col,
            )
            n_para = int(current.get("n_fallback_applied", 0))
            fb_total.extend(list(current.get("fallback_applied_pairs") or []))
            print(f"[{name}] paragraph freetext: n_fallback_applied={n_para}")

        if has_pp:
            per_pair = json.loads(per_pair_path.read_text(encoding="utf-8"))
            if not isinstance(per_pair, dict):
                raise TypeError(f"Expected object in {per_pair_path}")
            current = merge_with_freetext_fallback(
                current,
                per_pair,
                column_set=col,
            )
            n_pp = int(current.get("n_fallback_applied", 0))
            fb_total.extend(list(current.get("fallback_applied_pairs") or []))
            print(f"[{name}] per-pair freetext: n_fallback_applied={n_pp}")

        current["n_fallback_applied"] = n_para + n_pp
        current["fallback_applied_pairs"] = sorted(
            fb_total,
            key=lambda p: (p[0], p[1]),
        )
        _atomic_write_json(out_path, current)
        print(f"[{name}] wrote {out_path} (n_fallback_applied total={n_para + n_pp})")


if __name__ == "__main__":
    main()
