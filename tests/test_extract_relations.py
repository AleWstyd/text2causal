import json
import unittest
from unittest.mock import MagicMock, patch


class TestExtractRelations(unittest.TestCase):
    @patch("llm.extract_relations._get_client")
    def test_calls_openrouter_free_and_parses_json(self, mock_get_client) -> None:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        payload = [
            {
                "cause": "A",
                "effect": "B",
                "confidence": 0.9,
                "relation_type": "required",
            }
        ]
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps(payload)))]
        )

        from llm.extract_relations import MODEL, extract_relations

        result = extract_relations(["A", "B"], "A causes B.")
        self.assertEqual(result, payload)
        mock_client.chat.completions.create.assert_called_once()
        _, kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(kwargs["model"], MODEL)

    @patch("llm.extract_relations._get_client")
    def test_parses_json_inside_markdown_fence(self, mock_get_client) -> None:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        payload = [
            {
                "cause": "A",
                "effect": "B",
                "confidence": 0.9,
                "relation_type": "required",
            }
        ]
        body = json.dumps(payload, indent=2)
        fenced = f"```json\n{body}\n```"
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=fenced))]
        )

        from llm.extract_relations import extract_relations

        self.assertEqual(extract_relations(["A", "B"], "t"), payload)

    @patch("llm.extract_relations._get_client")
    def test_parses_fence_after_leading_text(self, mock_get_client) -> None:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        payload: list[dict] = []
        fenced = f"Here is the data.\n```json\n{json.dumps(payload)}\n```"
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=fenced))]
        )

        from llm.extract_relations import extract_relations

        self.assertEqual(extract_relations(["A"], "t"), payload)

    @patch("llm.extract_relations._get_client")
    @patch("builtins.print")
    def test_invalid_json_returns_empty(self, _mock_print, mock_get_client) -> None:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="not json"))]
        )

        from llm.extract_relations import extract_relations

        self.assertEqual(extract_relations(["X"], "t"), [])
