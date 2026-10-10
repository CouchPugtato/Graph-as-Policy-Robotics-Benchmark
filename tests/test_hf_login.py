import pytest

from gap_isaac.hf_login import clean_token


def test_hidden_input_normalized():
    assert clean_token("  hf_test_value\r\n") == "hf_test_value"


@pytest.mark.parametrize("value", ["", " ", "expired-oauth", "hf_one hf_two", "hf_one\nother"])
def test_no_empty_input_or_cached_fallback(value):
    with pytest.raises(ValueError, match="personal access token"):
        clean_token(value)
