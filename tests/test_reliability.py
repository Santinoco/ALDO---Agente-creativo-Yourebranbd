import pytest

from yourebrand.core.errors import (
    InvalidRequestError,
    ProviderTimeoutError,
    RateLimitError,
    TransientProviderError,
)
from yourebrand.core.reliability import InvalidUserInputError, call_with_retry, validate_user_text


def test_valid_text_is_returned_trimmed():
    assert validate_user_text("  hacela más corta\n") == "hacela más corta"


def test_short_text_is_valid():
    assert validate_user_text("sí") == "sí"


def test_text_keeps_line_breaks_and_tabs():
    assert validate_user_text("primera línea\nsegunda\tlínea") == "primera línea\nsegunda\tlínea"


@pytest.mark.parametrize("text", [None, 123, b"hola", ["hola"]])
def test_non_string_fails(text):
    with pytest.raises(InvalidUserInputError):
        validate_user_text(text)


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_blank_text_fails(text):
    with pytest.raises(InvalidUserInputError):
        validate_user_text(text)


def test_text_at_the_limit_is_valid_and_one_more_fails():
    assert validate_user_text("a" * 10, max_len=10) == "a" * 10

    with pytest.raises(InvalidUserInputError):
        validate_user_text("a" * 11, max_len=10)


@pytest.mark.parametrize("char", ["\x00", "\x07", "\x1b", "\x7f"])
def test_control_characters_fail(char):
    with pytest.raises(InvalidUserInputError):
        validate_user_text(f"hola{char}mundo")


def test_error_does_not_quote_the_text():
    with pytest.raises(InvalidUserInputError) as error:
        validate_user_text("dato confidencial\x00")

    assert "dato confidencial" not in str(error.value)


class Flaky:
    """Falla con los errores dados, en orden, y después devuelve "ok"."""

    def __init__(self, *errors):
        self.errors = list(errors)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return "ok"


def retry(fn, **overrides):
    """Reintenta sin dormir de verdad y devuelve también las esperas pedidas."""
    delays = []
    options = {"max_retries": 2, "jitter": False, "sleep": delays.append, **overrides}
    return call_with_retry(fn, **options), delays


def test_success_on_first_attempt_does_not_wait():
    fn = Flaky()

    (result, attempts), delays = retry(fn)

    assert (result, attempts) == ("ok", 1)
    assert delays == []


def test_retries_transient_errors_until_success():
    fn = Flaky(TransientProviderError("caído"), ProviderTimeoutError("lento"))

    (result, attempts), delays = retry(fn)

    assert (result, attempts) == ("ok", 3)
    assert fn.calls == 3


def test_delay_grows_exponentially():
    fn = Flaky(TransientProviderError("1"), TransientProviderError("2"), TransientProviderError("3"))

    _, delays = retry(fn, max_retries=3, base_delay_seconds=1.0)

    assert delays == [1.0, 2.0, 4.0]


def test_delay_is_capped():
    fn = Flaky(TransientProviderError("1"), TransientProviderError("2"), TransientProviderError("3"))

    _, delays = retry(fn, max_retries=3, base_delay_seconds=1.0, max_delay_seconds=1.5)

    assert delays == [1.0, 1.5, 1.5]


def test_jitter_stays_close_to_the_base_delay():
    fn = Flaky(TransientProviderError("1"))

    _, delays = retry(fn, jitter=True, base_delay_seconds=1.0)

    assert 0.8 <= delays[0] <= 1.2


def test_raises_last_error_when_retries_are_exhausted():
    fn = Flaky(TransientProviderError("1"), TransientProviderError("2"), ProviderTimeoutError("3"))

    with pytest.raises(ProviderTimeoutError):
        retry(fn, max_retries=2)

    assert fn.calls == 3


def test_zero_retries_makes_a_single_attempt():
    fn = Flaky(TransientProviderError("caído"))

    with pytest.raises(TransientProviderError):
        retry(fn, max_retries=0)

    assert fn.calls == 1


@pytest.mark.parametrize("error", [InvalidRequestError("mal armado"), ValueError("otro")])
def test_non_transient_errors_are_not_retried(error):
    fn = Flaky(error)

    with pytest.raises(type(error)):
        retry(fn)

    assert fn.calls == 1


def test_rate_limit_waits_what_the_provider_asks():
    fn = Flaky(RateLimitError("límite", retry_after_seconds=7))

    _, delays = retry(fn)

    assert delays == [7]


def test_rate_limit_wait_is_capped():
    fn = Flaky(RateLimitError("límite", retry_after_seconds=600))

    _, delays = retry(fn, max_delay_seconds=30.0)

    assert delays == [30.0]


def test_rate_limit_without_retry_after_uses_backoff():
    fn = Flaky(RateLimitError("límite"))

    _, delays = retry(fn, base_delay_seconds=1.0)

    assert delays == [1.0]


def test_reports_each_retry():
    events = []
    fn = Flaky(RateLimitError("límite", retry_after_seconds=3), TransientProviderError("caído"))

    retry(fn, on_retry=events.append, base_delay_seconds=1.0)

    assert [(event.attempt, event.error_type, event.delay_seconds) for event in events] == [
        (1, "RateLimitError", 3),
        (2, "TransientProviderError", 2.0),
    ]
