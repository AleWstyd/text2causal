from constraints.constraint_builder import build_constraints
from causal_discovery.run_pc import run_pc
from causal_discovery.run_ges import run_ges
from causal_discovery.run_lingam import run_lingam
from llm.extract_relations import extract_relations
from utils.graph_utils import apply_constraints, visualize_graph
from utils.load_data import load_lucas_dataset
from evaluation.metrics import precision, recall, f1_score


TRUE_EDGES = [
    ("Smoking", "Yellow_Fingers"),
    ("Anxiety", "Smoking"),
    ("Peer_Pressure", "Smoking"),
    ("Smoking", "Lung_Cancer"),
    ("Lung_Cancer", "Coughing"),
    ("Lung_Cancer", "Fatigue"),
    ("Allergy", "Coughing"),
    ("Coughing", "Fatigue"),
    ("Genetics", "Lung_Cancer"),
    ("Genetics", "Attention_Disorder"),
    ("Fatigue", "Car_Accident"),
    ("Attention_Disorder", "Car_Accident"),
]

# -----------------------------
# Load dataset
# -----------------------------

data = load_lucas_dataset("data/lucas")

variable_names = list(data.columns)

X = data.values


# -----------------------------
# Load background text
# -----------------------------

with open("data/lucas/lucas_background.txt") as f:
    background_text = f.read()


# -----------------------------
# LLM extraction
# -----------------------------

relations = extract_relations(variable_names, background_text)

print("LLM relations:")
print(relations)


required_edges = build_constraints(relations)

print("Constraints:")
print(required_edges)


# -----------------------------
# BASELINE MODELS
# -----------------------------

print("Running baseline PC")

pc_graph = run_pc(X, variable_names)

visualize_graph(pc_graph, "PC Baseline")


print("Running baseline GES")

ges_graph = run_ges(X, variable_names)

visualize_graph(ges_graph, "GES Baseline")


print("Running baseline LiNGAM")

lingam_graph = run_lingam(X, variable_names)

visualize_graph(lingam_graph, "LiNGAM Baseline")


# -----------------------------
# CONSTRAINT-GUIDED MODELS
# -----------------------------

pc_constrained = apply_constraints(pc_graph.copy(), required_edges)

ges_constrained = apply_constraints(ges_graph.copy(), required_edges)

lingam_constrained = apply_constraints(lingam_graph.copy(), required_edges)


visualize_graph(pc_constrained, "PC + LLM")

visualize_graph(ges_constrained, "GES + LLM")

visualize_graph(lingam_constrained, "LiNGAM + LLM")


# -----------------------------
# Evaluation
# (example placeholder)
# -----------------------------

pc_edges = list(pc_graph.edges())

p = precision(pc_edges, TRUE_EDGES)

r = recall(pc_edges, TRUE_EDGES)

f = f1_score(p, r)

print("PC Precision:", p)
print("PC Recall:", r)
print("PC F1:", f)
