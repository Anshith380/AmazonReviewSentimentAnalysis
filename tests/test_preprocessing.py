
import sys
from pathlib import Path

# Make the src folder importable during testing.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import clean_text


def test_clean_text_converts_to_lowercase():
    assert clean_text("GREAT Product") == "great product"


def test_clean_text_removes_html():
    assert clean_text("<p>Excellent product</p>") == "excellent product"


def test_clean_text_removes_url():
    result = clean_text("Good product https://example.com")
    assert "https://" not in result
    assert "good product" in result


def test_clean_text_normalizes_whitespace():
    assert clean_text("Good   product\n today") == "good product today"


def test_clean_text_preserves_negation():
    assert "not good" in clean_text("NOT GOOD")


def test_clean_text_handles_missing_value():
    assert clean_text(None) == ""


def test_clean_text_handles_empty_text():
    assert clean_text("   ") == ""
