import json

from llm.client import MODEL, _get_client
from llm.prompts import RELATION_PROMPT


def _unwrap_json_text(raw: str) -> str:
    """Strip optional ``` / ```json fences; models often wrap JSON despite the prompt."""
    s = raw.strip()
    if "```" not in s:
        return s
    start = s.find("```")
    rest = s[start + 3 :]
    if rest.lower().startswith("json"):
        rest = rest[4:].lstrip()
    else:
        rest = rest.lstrip()
    if rest.rstrip().endswith("```"):
        rest = rest.rstrip()[:-3]
    return rest.strip()


def extract_relations(variables, text):
    prompt = RELATION_PROMPT.format(variables=", ".join(variables), text=text)

    # Prompt asks for a JSON array; `json_object` response_format only allows top-level objects.
    response = _get_client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = (response.choices[0].message.content or "").strip()
    to_parse = _unwrap_json_text(raw)

    try:
        relations = json.loads(to_parse)
    except Exception as e:
        print("JSON parsing error:", e)
        print(raw)
        relations = []

    return relations
