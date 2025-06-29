import pytest
import pandas as pd
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analyzer import (
    sanitize_input,
    translate_to_english,
    analyze_sentiment_vader,
    analyze_sentiment_bert,
    generate_wordcloud,
    load_department_model,
    predict_department,
    process_feedback,
)
from langdetect.lang_detect_exception import LangDetectException

# Mock for translation to avoid hitting API during tests
if __name__ == "__main__":
    print(analyze_sentiment_bert("I love this movie."))   # expect Positive
    print(analyze_sentiment_bert("I hate this movie."))   # expect Negative


class DummyTranslator:
    def __init__(self, source, target):
        pass
    def translate(self, text):
        # Fake translation: append "[translated]"
        return text + " [translated]"

# Patch GoogleTranslator for tests
import analyzer
analyzer.GoogleTranslator = DummyTranslator

# ------------------- UNIT TESTS -------------------

def test_sanitize_input():
    raw_text = '<script>alert("XSS")</script>'
    sanitized = sanitize_input(raw_text)
    assert sanitized == '&lt;script&gt;alert(&quot;XSS&quot;)&lt;/script&gt;'


def test_translate_to_english_english_text():
    # English text should return as is
    text = "This is a test."
    result = translate_to_english(text)
    assert result == text


def test_translate_to_english_non_english_text(monkeypatch):
    # Non-English text triggers translation (fake)
    non_english_text = "Bonjour"
    # monkeypatch langdetect.detect to return 'fr'
    monkeypatch.setattr(analyzer, "detect", lambda x: 'fr')
    result = translate_to_english(non_english_text)
    assert result.endswith("[translated]")


def test_translate_to_english_exception(monkeypatch):
    # If detect throws, function returns input text
    def raise_exception(_):
        raise LangDetectException("error")
    monkeypatch.setattr(analyzer, "detect", raise_exception)
    text = "some text"
    result = translate_to_english(text)
    assert result == text


def test_analyze_sentiment_vader():
    pos_text = "I love this product!"
    neg_text = "This is terrible."
    

    assert analyze_sentiment_vader(pos_text) == "Positive"
    assert analyze_sentiment_vader(neg_text) == "Negative"



def test_analyze_sentiment_bert():
    pos_text = "I love this movie."
    neg_text = "I hate this movie."
    

    print(f"pos_text: {pos_text!r}")
    print(f"neg_text: {neg_text!r}")
   

    print("Positive:", analyze_sentiment_bert(pos_text))
    print("Negative:", analyze_sentiment_bert(neg_text))
    

    assert analyze_sentiment_bert(pos_text) == "Positive"
    assert analyze_sentiment_bert(neg_text) == "Negative"


def test_generate_wordcloud():
    feedback_series = pd.Series(["test feedback", "more feedback"])
    wc = generate_wordcloud(feedback_series)
    assert wc is not None
    assert hasattr(wc, "generate")


@pytest.fixture
def dummy_model():
    class DummyModel:
        def predict(self, texts):
            return ["HR" for _ in texts]
    return DummyModel()

def test_predict_department(dummy_model):
    feedback = "This is about benefits."
    dept = predict_department(feedback, dummy_model)
    assert dept == "HR"


def test_process_feedback_basic(monkeypatch):
    # Patch translate_to_english to bypass API
    monkeypatch.setattr(analyzer, "translate_to_english", lambda x: x)
    df = pd.DataFrame({
        "feedback": [
            "Good product",
            "Bad service",
            None,
            "<script>bad</script>"
        ]
    })
    result_df = process_feedback(df, "feedback", use_bert=False)
    assert "Sentiment" in result_df.columns
    assert all(s in ["Positive", "Negative"] for s in result_df["Sentiment"])
    # Check sanitization applied
    assert "&lt;script&gt;" in result_df.loc[3, "feedback"]


def test_process_feedback_use_bert(monkeypatch):
    monkeypatch.setattr(analyzer, "translate_to_english", lambda x: x)
    df = pd.DataFrame({"feedback": ["Good product", "Bad product"]})
    result_df = process_feedback(df, "feedback", use_bert=True)
    assert "Sentiment" in result_df.columns
    assert all(s in ["Positive", "Negative"] for s in result_df["Sentiment"])
    

# ------------------- INTEGRATION TEST EXAMPLES -------------------

def test_end_to_end_feedback_processing(monkeypatch):
    monkeypatch.setattr(analyzer, "translate_to_english", lambda x: x)
    monkeypatch.setattr(analyzer, "sanitize_input", lambda x: x)
    df = pd.DataFrame({"feedback": ["I love this!", "Je déteste ça.", "Okay experience"]})
    result_df = process_feedback(df, "feedback")
    # We expect the second row to be translated (mock disables translation here)
    assert "Sentiment" in result_df.columns
    assert len(result_df) == 3



@pytest.fixture()
def mock_pipeline():
    with patch('analyzer.pipeline') as mock_pipeline_fn:
        mock_pipe_instance = MagicMock()
        mock_pipe_instance.return_value = [{'label': 'POSITIVE', 'score': 0.99}]
        mock_pipeline_fn.return_value = mock_pipe_instance
        yield
