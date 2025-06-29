# Feedback Analyzer

**Feedback Analyzer** is a Python-based web application designed to analyze customer feedback using Natural Language Processing (NLP) and Machine Learning. It classifies feedback by **sentiment** and **department**, providing meaningful visualizations and exports for improved organizational insight.

##  Features

- Upload and analyze customer feedback from CSV files.
- Department classification using a pre-trained ML model.
- Sentiment analysis using VADER.
- Interactive visualizations (time-based, sentiment trend, department breakdown).
- Export categorized feedback as CSV.
- Logging for debugging and monitoring.
- Automated testing with Pytest.

##  Tech Stack

- **Backend:** streamlit (via `app.py`)
- **ML/NLP:** scikit-learn, VADER SentimentIntensityAnalyzer
- **Data Processing:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Testing:** Pytest
- **Model Storage:** Pickle (`department_model.pkl`)

##  Project Structure

```
feedback analyzer code/
│
├── app.py                        # Main Flask app
├── analyzer.py                  # Feedback analysis logic
├── department_model.pkl         # Pre-trained department classifier
├── department_feedback.csv      # Sample input CSV
├── requirements.txt             # Python dependencies
├── pytest.ini                   # Pytest configuration
├── test_log.log                 # Test log output
│
├── tests/                       # Unit and integration tests
│   ├── test_app_suite.py
│   └── test_ui.py
│
└── .pytest_cache/               # Pytest cache directory
```

##  Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/feedback-analyzer.git
cd feedback-analyzer
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
streamlit run app.py
```

Access the app at: [http://127.0.0.1:5000](http://127.0.0.1:5000)

### 4. Run Tests

```bash
pytest
```

##  Sample Input

Ensure your CSV file (`department_feedback.csv`) follows this format:

```csv
Feedback
"The checkout process was too slow."
"Customer support was very helpful!"
...
```

##  Output

- Categorized and sentiment-tagged feedback
- Exported CSV file for download
- Visualization dashboard in browser

##  Notes

- The department classification model (`department_model.pkl`) is already trained.
- Make sure input data is cleaned (no null values) for optimal results.

##  License

This project is licensed under the MIT License.
