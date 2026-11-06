"""Behavior checks beyond the supplied grading examples; no network calls."""

from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from test_template import _m


@pytest.mark.parametrize(
    "target,function,method",
    [
        ("openai.OpenAI", _m.call_openai, "chat.completions.create"),
        ("google.genai.Client", _m.call_gemini, "models.generate_content"),
        ("anthropic.Anthropic", _m.call_anthropic, "messages.create"),
    ],
)
def test_api_errors_propagate_and_clients_close(target, function, method):
    with patch(target) as factory:
        endpoint = factory.return_value
        for attribute in method.split("."):
            endpoint = getattr(endpoint, attribute)
        error = RuntimeError("provider unavailable")
        endpoint.side_effect = error
        with pytest.raises(RuntimeError) as caught:
            function("Hello")
        assert caught.value is error
        factory.return_value.close.assert_called_once_with()


def test_openai_forwards_parameters_and_reads_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")
    with patch("openai.OpenAI") as factory:
        response = factory.return_value.chat.completions.create.return_value
        response.choices = [SimpleNamespace(message=SimpleNamespace(content=None))]
        response.usage = SimpleNamespace(prompt_tokens=2, completion_tokens=0)
        text, latency, usage = _m.call_openai("Hi", "custom-model", 0.2, 0.8, 42)
        factory.assert_called_once_with(api_key="test-placeholder")
        factory.return_value.chat.completions.create.assert_called_once_with(
            model="custom-model", messages=[{"role": "user", "content": "Hi"}],
            temperature=0.2, top_p=0.8, max_tokens=42,
        )
        assert text == ""
        assert latency >= 0
        assert usage == {"input_tokens": 2, "output_tokens": 0}


def test_gemini_forwards_generation_config():
    with patch("google.genai.Client") as factory:
        response = factory.return_value.models.generate_content.return_value
        response.text = "Xin chào"
        response.usage_metadata = SimpleNamespace(
            prompt_token_count=4, candidates_token_count=5
        )
        assert _m.call_gemini("Hi", "custom-model", 0.2, 0.8, 42)[0] == "Xin chào"
        kwargs = factory.return_value.models.generate_content.call_args.kwargs
        assert kwargs["model"] == "custom-model"
        assert kwargs["contents"] == "Hi"
        assert kwargs["config"].temperature == 0.2
        assert kwargs["config"].top_p == 0.8
        assert kwargs["config"].max_output_tokens == 42


def test_anthropic_collects_text_blocks_and_forwards_parameters():
    with patch("anthropic.Anthropic") as factory:
        response = factory.return_value.messages.create.return_value
        response.content = [
            SimpleNamespace(text="Hello "),
            SimpleNamespace(type="tool_use"),
            SimpleNamespace(text="world"),
        ]
        response.usage = SimpleNamespace(input_tokens=1, output_tokens=2)
        text, _, usage = _m.call_anthropic("Hi", "custom-model", 0.2, 0.8, 42)
        assert text == "Hello world"
        assert usage == {"input_tokens": 1, "output_tokens": 2}
        factory.return_value.messages.create.assert_called_once_with(
            model="custom-model", messages=[{"role": "user", "content": "Hi"}],
            temperature=0.2, top_p=0.8, max_tokens=42,
        )


def test_retry_uses_exact_delays_and_preserves_last_exception():
    error = RuntimeError("still unavailable")
    function = MagicMock(side_effect=error)
    with patch.object(_m.time, "sleep") as sleep:
        with pytest.raises(RuntimeError) as caught:
            _m.retry_with_backoff(function, max_retries=3, base_delay=0.1)
    assert caught.value is error
    assert function.call_count == 4
    assert sleep.call_args_list == [call(0.1), call(0.2), call(0.4)]


def test_zero_retries_makes_one_attempt_without_sleeping():
    function = MagicMock(side_effect=ValueError("fail"))
    with patch.object(_m.time, "sleep") as sleep:
        with pytest.raises(ValueError):
            _m.retry_with_backoff(function, max_retries=0)
    function.assert_called_once_with()
    sleep.assert_not_called()


@pytest.mark.parametrize("kwargs", [{"max_retries": -1}, {"base_delay": -0.1}])
def test_negative_retry_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        _m.retry_with_backoff(lambda: None, **kwargs)


def test_streaming_preserves_three_complete_turns(capsys):
    with patch("google.genai.Client") as factory, patch(
        "builtins.input", side_effect=["", "q1", "q2", "q3", "q4", "q5", " EXIT "]
    ):
        stream = factory.return_value.models.generate_content_stream
        stream.side_effect = [
            [SimpleNamespace(text=None), SimpleNamespace(text="hello"), SimpleNamespace(text="!")]
            for _ in range(5)
        ]
        _m.streaming_chatbot()
    requests = [item.kwargs["contents"] for item in stream.call_args_list]
    assert len(requests) == 5
    assert [len(contents) for contents in requests] == [1, 3, 5, 7, 7]
    assert [message.role for message in requests[-1]] == [
        "user", "model", "user", "model", "user", "model", "user"
    ]
    assert [message.parts[0].text for message in requests[-1]] == [
        "q2", "hello!", "q3", "hello!", "q4", "hello!", "q5"
    ]
    assert capsys.readouterr().out.count("Gemini: hello!") == 5
    factory.return_value.close.assert_called_once_with()


@pytest.mark.parametrize("error", [EOFError, KeyboardInterrupt])
def test_streaming_handles_terminal_shutdown(error):
    with patch("google.genai.Client") as factory, patch("builtins.input", side_effect=error):
        _m.streaming_chatbot()
    factory.return_value.models.generate_content_stream.assert_not_called()
    factory.return_value.close.assert_called_once_with()


def test_empty_stream_does_not_add_incomplete_history():
    with patch("google.genai.Client") as factory, patch(
        "builtins.input", side_effect=["q1", "q2", "quit"]
    ):
        stream = factory.return_value.models.generate_content_stream
        stream.side_effect = [[], [SimpleNamespace(text="answer")]]
        _m.streaming_chatbot()
    assert len(stream.call_args_list[1].kwargs["contents"]) == 1


def test_batch_forwards_prompts_without_mutating_comparison():
    comparison = {"gpt4o": {"response": "answer"}}
    with patch.object(_m, "compare_models", return_value=comparison) as compare:
        results = _m.batch_compare(["Q1", "Q2"])
        assert compare.call_args_list == [call("Q1"), call("Q2")]
        assert [result["prompt"] for result in results] == ["Q1", "Q2"]
        assert "prompt" not in comparison
        assert _m.batch_compare([]) == []


def test_markdown_escapes_pipes_newlines_and_truncates_responses():
    table = _m.format_comparison_table([{
        "prompt": "First | line\nsecond line",
        "gpt4o": {
            "response": "answer |\n" + "x" * 100,
            "latency": 0.1234, "input_tokens": 10, "output_tokens": 20,
            "cost": 0.00000675,
        },
    }])
    lines = table.splitlines()
    assert len(lines) == 3
    assert all(line.count("|") == 7 for line in lines)
    assert "First &#124; line second line" in table
    response = lines[2].split("|")[3].strip().replace("&#124;", "|")
    assert len(response) == 50
    assert response.endswith("...")
    assert "$0.00000675" in table
    assert len(_m.format_comparison_table([]).splitlines()) == 2
