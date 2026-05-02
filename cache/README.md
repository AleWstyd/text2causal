# Response Cache

This directory stores replayable HTTP and LLM responses as one JSON file per
request key.

- `cache/llm/`: OpenRouter chat completion responses.
- `cache/reactome/`: Reactome REST responses, added in Step 2.

Keys are SHA256 hashes of the JSON request payload with sorted keys. LLM cache
entries store the full response dict; `model` is the OpenRouter-served model
used for appendix reporting.

Committed cache entries make later experiment runs reproducible without API
access when every requested payload already has a cache file.
