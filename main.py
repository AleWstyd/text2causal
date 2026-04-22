from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st

from app_pipeline import (
    extract_llm_constraints,
    get_algorithm_names,
    load_app_inputs,
    run_algorithm_simulation,
)
from constraints.constraint_builder import PriorKnowledge
from utils.graph_utils import build_graph_figure, build_hierarchical_layout


st.set_page_config(page_title="Text2Causal", layout="wide")


@st.cache_data(show_spinner=False)
def get_inputs() -> dict[str, object]:
    return load_app_inputs()


def initialize_session_state() -> None:
    st.session_state.setdefault("llm_state", None)
    st.session_state.setdefault("llm_state_background_text", None)
    st.session_state.setdefault("run_results", {})


def ensure_llm_state(
    variable_names: list[str], background_text: str
) -> dict[str, object]:
    needs_refresh = (
        st.session_state["llm_state"] is None
        or st.session_state["llm_state_background_text"] != background_text
    )

    if needs_refresh:
        with st.spinner("Extracting relations with the LLM (OpenRouter)..."):
            st.session_state["llm_state"] = extract_llm_constraints(
                variable_names=variable_names,
                background_text=background_text,
            )
            st.session_state["llm_state_background_text"] = background_text

    return st.session_state["llm_state"]


def build_relations_frame(relations: list[dict[str, object]]) -> pd.DataFrame:
    if not relations:
        return pd.DataFrame(columns=["relation_type", "cause", "effect", "confidence"])

    normalized_relations = [
        {
            "relation_type": relation.get("relation_type", "required"),
            "cause": relation.get("cause"),
            "effect": relation.get("effect"),
            "confidence": relation.get("confidence"),
        }
        for relation in relations
    ]
    frame = pd.DataFrame(
        normalized_relations,
        columns=["relation_type", "cause", "effect", "confidence"],
    )

    if "confidence" in frame:
        frame["confidence"] = frame["confidence"].astype(float).round(3)

    return frame


def build_constraints_frame(prior_knowledge: PriorKnowledge) -> pd.DataFrame:
    rows = [
        {"type": "required", "cause": cause, "effect": effect}
        for cause, effect in prior_knowledge.required_edges
    ]
    rows.extend(
        {"type": "forbidden", "cause": cause, "effect": effect}
        for cause, effect in prior_knowledge.forbidden_edges
    )

    return pd.DataFrame(rows, columns=["type", "cause", "effect"])


def describe_constraint_mode(constraint_mode: str) -> str:
    descriptions = {
        "native_pc_background_knowledge": (
            "PC uses causallearn BackgroundKnowledge to orient edges when the learned "
            "skeleton allows them."
        ),
        "native_direct_lingam_prior_knowledge": (
            "LiNGAM uses DirectLiNGAM prior knowledge, where required and forbidden "
            "entries describe directed-path constraints."
        ),
        "post_hoc_direct_edge_constraints": (
            "GES has no native prior-knowledge hook in this causallearn version, so "
            "constraints are applied after graph discovery as direct-edge edits."
        ),
    }

    return descriptions[constraint_mode]


def build_metrics_frame(result: dict[str, object]) -> pd.DataFrame:
    baseline_metrics = result["baseline_metrics"]
    constrained_metrics = result["constrained_metrics"]

    return pd.DataFrame(
        [
            {"graph": "Unconstrained", **baseline_metrics},
            {"graph": "Prior knowledge", **constrained_metrics},
        ]
    ).round(3)


def render_graphs(result: dict[str, object]) -> None:
    combined_graph = nx.compose(result["baseline_graph"], result["constrained_graph"])
    layout = build_hierarchical_layout(combined_graph)

    left_column, right_column = st.columns(2)

    with left_column:
        st.subheader("Unconstrained result")
        baseline_figure = build_graph_figure(
            result["baseline_graph"],
            title="Unconstrained result",
            pos=layout,
        )
        st.pyplot(baseline_figure, width="stretch")
        plt.close(baseline_figure)

    with right_column:
        st.subheader("Prior-knowledge result")
        constrained_figure = build_graph_figure(
            result["constrained_graph"],
            title="Prior-knowledge result",
            pos=layout,
        )
        st.pyplot(constrained_figure, width="stretch")
        plt.close(constrained_figure)


def render_result(result: dict[str, object], llm_state: dict[str, object]) -> None:
    relations_frame = build_relations_frame(llm_state["relations"])
    constraints_frame = build_constraints_frame(llm_state["prior_knowledge"])

    left_column, middle_column, right_column = st.columns([1, 0.12, 1])

    with left_column:
        st.subheader("LLM-found edges")
        if relations_frame.empty:
            st.info("The LLM did not return any relations.")
        else:
            st.dataframe(relations_frame, width="stretch", hide_index=True)

    with middle_column:
        st.markdown(
            "<div style='text-align:center; font-size:2rem; padding-top:4.5rem;'>→</div>",
            unsafe_allow_html=True,
        )

    with right_column:
        st.subheader("Prior knowledge")
        if constraints_frame.empty:
            st.info("No relations met the confidence threshold for constraints.")
        else:
            st.dataframe(constraints_frame, width="stretch", hide_index=True)

    st.caption(describe_constraint_mode(result["constraint_mode"]))

    st.subheader("Metrics")
    st.dataframe(build_metrics_frame(result), width="stretch", hide_index=True)

    render_graphs(result)


def render_algorithm_tab(
    algorithm_name: str,
    inputs: dict[str, object],
) -> None:
    description_column, action_column = st.columns(
        [1, 0.24], vertical_alignment="bottom"
    )

    with description_column:
        st.write(f"Run the {algorithm_name} simulation on the Lucas dataset.")

    run_requested = False
    with action_column:
        run_requested = st.button(
            "Run simulation", key=f"run_{algorithm_name}", width="stretch"
        )

    background_text = st.text_area(
        "Simulation text",
        value=inputs["background_text"],
        height=180,
        key=f"background_text_{algorithm_name}",
    )

    if run_requested:
        try:
            llm_state = ensure_llm_state(
                variable_names=inputs["variable_names"],
                background_text=background_text,
            )
        except Exception as exc:
            st.session_state["run_results"].pop(algorithm_name, None)
            st.error(f"LLM extraction failed: {exc}")
        else:
            try:
                with st.spinner(f"Running {algorithm_name}..."):
                    result = run_algorithm_simulation(
                        algorithm_name=algorithm_name,
                        data_matrix=inputs["data_matrix"],
                        variable_names=inputs["variable_names"],
                        prior_knowledge=llm_state["prior_knowledge"],
                    )
            except Exception as exc:
                st.session_state["run_results"].pop(algorithm_name, None)
                st.error(f"{algorithm_name} failed: {exc}")
            else:
                result["background_text"] = background_text
                result["llm_state"] = llm_state
                st.session_state["run_results"][algorithm_name] = result

    stored_result = st.session_state["run_results"].get(algorithm_name)

    if (
        stored_result
        and stored_result.get("llm_state")
        and stored_result.get("background_text") == background_text
    ):
        render_result(stored_result, stored_result["llm_state"])


def main() -> None:
    initialize_session_state()
    inputs = get_inputs()

    title_column, summary_column = st.columns([0.9, 3.1], vertical_alignment="center")

    with title_column:
        st.markdown("<h1 style='margin:0;'>Text2Causal</h1>", unsafe_allow_html=True)

    with summary_column:
        st.caption(
            "Run PC, GES, and LiNGAM with shared LLM-extracted prior knowledge and compare "
            "unconstrained vs prior-knowledge-guided results."
        )

    tabs = st.tabs(get_algorithm_names())

    for tab, algorithm_name in zip(tabs, get_algorithm_names(), strict=False):
        with tab:
            render_algorithm_tab(algorithm_name, inputs)


if __name__ == "__main__":
    main()
