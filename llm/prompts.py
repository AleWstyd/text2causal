RELATION_PROMPT = """
You are an expert in causal reasoning.

Given the following variables:

{variables}

From the text below extract causal relationships between these variables.
Extract both:
- required edges, when the text states or strongly implies one variable causes another
- forbidden edges, when the text states a variable does not cause or is not caused by another

Treat negative statements such as "never", "does not", "is not caused by", and
"cannot cause" as forbidden edges.

Return ONLY valid JSON with no additional text.

Format:

[
  {{
    "cause": "variable_name",
    "effect": "variable_name",
    "confidence": 0.0-1.0,
    "relation_type": "required" | "forbidden"
  }}
]

Only use variables from the provided list.
Use exactly "required" or "forbidden" for relation_type.

Text:

{text}
"""
