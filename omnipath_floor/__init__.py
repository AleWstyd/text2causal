"""OmniPath direct-edge floor priors (Step 4 C0.5 ablation).

The local package is named ``omnipath_floor`` rather than ``omnipath`` to
avoid shadowing the upstream PyPI ``omnipath`` REST client which is the
data source we read from. ``docs/step_04_causal_reasoning.md`` records this
implementation deviation from the original ``omnipath/floor.py`` path
suggested in ``docs/dev_plan_v2.md`` §9.
"""

from omnipath_floor.floor import (
    FloorEdge,
    build_floor_priors,
    fetch_directed_interactions,
)

__all__ = ["FloorEdge", "build_floor_priors", "fetch_directed_interactions"]
