from __future__ import annotations

import unittest

import httpx

from reactome.client import _fetch_with_retry


def _response(status_code: int, payload: dict[str, object]) -> httpx.Response:
    request = httpx.Request("GET", "https://reactome.test/data")
    return httpx.Response(status_code, json=payload, request=request)


class FetchWithRetryTests(unittest.TestCase):
    def test_retries_retry_after_5xx_then_returns_json(self) -> None:
        responses = [
            httpx.Response(
                503,
                headers={"Retry-After": "1"},
                request=httpx.Request("GET", "https://reactome.test/data"),
            ),
            _response(200, {"ok": True}),
        ]
        sleeps: list[float] = []

        def method() -> httpx.Response:
            return responses.pop(0)

        result = _fetch_with_retry(method, sleep=sleeps.append)

        self.assertEqual(result, {"ok": True})
        self.assertEqual(len(sleeps), 1)
        self.assertGreaterEqual(sleeps[0], 1.0)

    def test_three_retryable_statuses_raise_status_error(self) -> None:
        responses = [
            _response(503, {"error": "one"}),
            _response(503, {"error": "two"}),
            _response(503, {"error": "three"}),
        ]

        def method() -> httpx.Response:
            return responses.pop(0)

        with self.assertRaises(httpx.HTTPStatusError):
            _fetch_with_retry(method, sleep=lambda _: None)

    def test_request_error_then_success_returns_json(self) -> None:
        request = httpx.Request("GET", "https://reactome.test/data")
        responses: list[httpx.Response | httpx.RequestError] = [
            httpx.RequestError("temporary", request=request),
            _response(200, {"ok": True}),
        ]
        sleeps: list[float] = []

        def method() -> httpx.Response:
            next_result = responses.pop(0)
            if isinstance(next_result, httpx.RequestError):
                raise next_result
            return next_result

        result = _fetch_with_retry(method, sleep=sleeps.append)

        self.assertEqual(result, {"ok": True})
        self.assertEqual(sleeps, [0.5])

    def test_three_request_errors_bubble_original_exception(self) -> None:
        request = httpx.Request("GET", "https://reactome.test/data")
        original = httpx.RequestError("boom", request=request)

        def method() -> httpx.Response:
            raise original

        with self.assertRaises(httpx.RequestError) as raised:
            _fetch_with_retry(method, sleep=lambda _: None)

        self.assertIs(raised.exception, original)


if __name__ == "__main__":
    unittest.main()
