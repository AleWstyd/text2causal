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
from utils.graph_utils import build_graph_figure, build_hierarchical_layout


st.set_page_config(page_title="Text2Causal", layout="wide")


@st.cache_data(show_spinner=False)
def get_inputs() -> dict[str, object]:
    return load_app_inputs()


def initialize_session_state() -> None:
    st.session_state.setdefault("llm_state", None)
    st.session_state.setdefault("run_results", {})


def ensure_llm_state(variable_names: list[str], background_text: str) -> dict[str, object]:
    if st.session_state["llm_state"] is None:
        with st.spinner("Extracting relations with Gemini..."):
            st.session_state["llm_state"] = extract_llm_constraints(
                variable_names=variable_names,
                background_text=background_text,
            )

    return st.session_state["llm_state"]


def build_relations_frame(relations: list[dict[str, object]]) -> pd.DataFrame:
    if not relations:
        return pd.DataFrame(columns=["cause", "effect", "confidence"])

    frame = pd.DataFrame(relations, columns=["cause", "effect", "confidence"])

    if "confidence" in frame:
        frame["confidence"] = frame["confidence"].astype(float).round(3)

    return frame


def build_constraints_frame(required_edges: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(required_edges, columns=["cause", "effect"])


def build_metrics_frame(result: dict[str, object]) -> pd.DataFrame:
    baseline_metrics = result["baseline_metrics"]
    constrained_metrics = result["constrained_metrics"]

    return pd.DataFrame(
        [
            {"graph": "Unconstrained", **baseline_metrics},
            {"graph": "LLM constrained", **constrained_metrics},
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
        st.pyplot(baseline_figure, use_container_width=True)
        plt.close(baseline_figure)

    with right_column:
        st.subheader("LLM constrained result")
        constrained_figure = build_graph_figure(
            result["constrained_graph"],
            title="LLM constrained result",
            pos=layout,
        )
        st.pyplot(constrained_figure, use_container_width=True)
        plt.close(constrained_figure)


def render_result(result: dict[str, object], llm_state: dict[str, object]) -> None:
    st.subheader("LLM-found edges")
    relations_frame = build_relations_frame(llm_state["relations"])
    if relations_frame.empty:
        st.info("The LLM did not return any relations.")
    else:
        st.dataframe(relations_frame, use_container_width=True, hide_index=True)

    st.subheader("Applied constraint edges")
    constraints_frame = build_constraints_frame(llm_state["required_edges"])
    if constraints_frame.empty:
        st.info("No relations met the confidence threshold for constraints.")
    else:
        st.dataframe(constraints_frame, use_container_width=True, hide_index=True)

    render_graphs(result)

    st.subheader("Metrics")
    st.dataframe(build_metrics_frame(result), use_container_width=True, hide_index=True)


def render_algorithm_tab(
    algorithm_name: str,
    inputs: dict[str, object],
) -> None:
    st.write(f"Run the {algorithm_name} simulation on the Lucas dataset.")

    if st.button("Run simulation", key=f"run_{algorithm_name}"):
        try:
            llm_state = ensure_llm_state(
                variable_names=inputs["variable_names"],
                background_text=inputs["background_text"],
            )
            with st.spinner(f"Running {algorithm_name}..."):
                result = run_algorithm_simulation(
                    algorithm_name=algorithm_name,
                    data_matrix=inputs["data_matrix"],
                    variable_names=inputs["variable_names"],
                    required_edges=llm_state["required_edges"],
                )
        except Exception as exc:
            st.session_state["run_results"].pop(algorithm_name, None)
            st.error(f"{algorithm_name} failed: {exc}")
        else:
            st.session_state["run_results"][algorithm_name] = result

    stored_result = st.session_state["run_results"].get(algorithm_name)
    llm_state = st.session_state["llm_state"]

    if stored_result and llm_state:
        render_result(stored_result, llm_state)


def main() -> None:
    initialize_session_state()
    inputs = get_inputs()

    st.title("Text2Causal")
    st.caption(
        "Run PC, GES, and LiNGAM with shared LLM-extracted causal edges and compare "
        "unconstrained vs LLM-constrained results."
    )

    tabs = st.tabs(get_algorithm_names())

    for tab, algorithm_name in zip(tabs, get_algorithm_names(), strict=False):
        with tab:
            render_algorithm_tab(algorithm_name, inputs)


if __name__ == "__main__":
    main()
