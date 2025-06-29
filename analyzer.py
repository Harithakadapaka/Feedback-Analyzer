import html
import joblib
from langdetect import detect
from deep_translator import GoogleTranslator
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from transformers import pipeline

# ---------- Input Sanitization ----------
def sanitize_input(text):
    return html.escape(str(text))

# ---------- Translation (Multilingual Support) ----------
def translate_to_english(text):
    try:
        lang = detect(text)
        if lang != 'en':
            return GoogleTranslator(source=lang, target='en').translate(text)
        return text
    except:
        return text  # In case detection or translation fails

# ---------- VADER Sentiment Analysis ----------
def analyze_sentiment_vader(feedback):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(feedback)
    if sentiment_score['compound'] >= 0.1:
        return 'Positive'
    else:
        return 'Negative'

# ---------- BERT Sentiment Analysis ----------


_bert_pipeline = None

def analyze_sentiment_bert(text):
    global _bert_pipeline
    if _bert_pipeline is None:
        _bert_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    try:
        # Pass input as list (batch) to avoid weird single string processing
        result = _bert_pipeline([text])[0]
        print(f"Input: {text} -> Label: {result['label']}, Score: {result['score']}")
        label = result['label']
        return 'Positive' if label == 'POSITIVE' else 'Negative'
    except Exception:
        return 'Negative'  # Fallback

# ---------- WordCloud Generation ----------
def generate_wordcloud(feedback_series):
    text = " ".join(feedback_series.fillna(''))
    wordcloud = WordCloud(width=800, height=400).generate(text)
    return wordcloud

# ---------- Department Classifier ----------
def load_department_model(path='department_model.pkl'):
    return joblib.load(path)

def predict_department(feedback, department_pipeline):
    return department_pipeline.predict([feedback])[0]

# ---------- Main Feedback Processor ----------
def process_feedback(df, text_column, use_bert=False):
    df[text_column] = df[text_column].fillna('').astype(str)

    # Translate to English if needed
    df[text_column] = df[text_column].apply(translate_to_english)

    # Sanitize input
    df[text_column] = df[text_column].apply(sanitize_input)

    # Sentiment analysis
    if use_bert:
        df['Sentiment'] = df[text_column].apply(analyze_sentiment_bert)
    else:
        df['Sentiment'] = df[text_column].apply(analyze_sentiment_vader)

    return df
