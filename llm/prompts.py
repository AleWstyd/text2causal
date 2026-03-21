RELATION_PROMPT = """
You are an expert in causal reasoning.

Given the following variables:

{variables}

From the text below extract causal relationships between these variables.

Return ONLY valid JSON with no additional text.

Format:

[
  {{
    "cause": "variable_name",
    "effect": "variable_name",
    "confidence": 0.0-1.0
  }}
]

Only use variables from the provided list.

Text:

{text}
"""
