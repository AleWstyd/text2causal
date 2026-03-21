import json
from google import genai
from llm.prompts import RELATION_PROMPT


MODEL = "gemini-3.1-flash-lite-preview"

client = genai.Client()


def extract_relations(variables, text):

    prompt = RELATION_PROMPT.format(variables=", ".join(variables), text=text)

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )

    try:
        relations = json.loads(response.text)
    except Exception as e:
        print("JSON parsing error:", e)
        print(response.text)
        relations = []

    return relations
