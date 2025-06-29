import streamlit as st
import pandas as pd
import joblib
import html
import numpy as np
import traceback
from io import StringIO, BytesIO
import datetime
import matplotlib.pyplot as plt
from langdetect import detect
from deep_translator import GoogleTranslator
from transformers import pipeline
from wordcloud import WordCloud
import plotly.express as px
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from cryptography.fernet import Fernet  # For encryption of downloads
import chardet 
import time


def create_sentiment_pie_chart(df):
    sentiment_counts = df['Sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    fig = px.pie(sentiment_counts, values='Count', names='Sentiment', title='Sentiment Share', hole=0.3)
    return fig

def main():

    start_time = time.time() 

    st.set_page_config(layout="wide", page_title="Feedback analyzer", page_icon="🧠")
    st.title("🧠 Feedback analyzer")
    # -------------------- Encryption Setup --------------------
    def generate_key():
        # In real app, store securely, here we generate per session for demo
        if 'fernet_key' not in st.session_state:
            st.session_state['fernet_key'] = Fernet.generate_key()
        return st.session_state['fernet_key']

    def encrypt_bytes(data_bytes):
        key = generate_key()
        fernet = Fernet(key)
        encrypted = fernet.encrypt(data_bytes)
        return encrypted

    def decrypt_bytes(encrypted_bytes):
        key = generate_key()
        fernet = Fernet(key)
        decrypted = fernet.decrypt(encrypted_bytes)
        return decrypted

    # -------------------- File Validators --------------------
    def validate_file(uploaded_file):
        if uploaded_file is None:
            raise ValueError("No file uploaded.")
        if uploaded_file.type != "text/csv":
            raise ValueError("Please upload a valid CSV file.")
        if uploaded_file.size > 10_000_000:  # ~10MB limit
            raise ValueError("File size exceeds 10MB limit.")

    @st.cache_data(show_spinner=False)
    def read_csv_safe(uploaded_file):
        try:
            uploaded_file.seek(0)
            rawdata = uploaded_file.read(10000)  # sample bytes for encoding detection
            uploaded_file.seek(0)

            result = chardet.detect(rawdata)
            encoding = result['encoding'] if result['encoding'] else 'utf-8'

            #st.write(f"Detected encoding: {encoding}")

            try:
                df = pd.read_csv(uploaded_file, encoding=encoding)
                if df.empty or df.columns.size == 0:
                    raise ValueError("CSV appears to have no columns or content.")
                return df
            except Exception as e:
                st.warning(f"Failed with encoding {encoding}: {e}")
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
                if df.empty or df.columns.size == 0:
                    raise ValueError("CSV appears to have no columns or content.")
                return df
        except Exception as e:
            raise ValueError(f"Failed to read CSV: {e}")

    # -------------------- Text Processing --------------------
    def translate_to_english(text):
        try:
            lang = detect(text)
            if lang != 'en':
                translated = GoogleTranslator(source=lang, target='en').translate(text)
                return translated
        except Exception:
            return text
        return text

    # -------------------- Sentiment Analysis --------------------
    @st.cache_resource(show_spinner=False)
    def get_sentiment_pipeline():
        return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", framework="pt")

    def analyze_sentiment_bert(text):
        # Rule-based override for upgrade/subscribe intent
        text_lower = text.lower()
        upgrade_keywords = ["upgrade", "premium", "subscribe", "subscription", "trial"]
        
        if any(word in text_lower for word in upgrade_keywords):
            return "POSITIVE"  # or "NEUTRAL" if you prefer a conservative choice

        # Use BERT sentiment pipeline if no override
        sentiment_pipeline = get_sentiment_pipeline()
        result = sentiment_pipeline(text)[0]
        return result['label'].upper()
    


    def process_feedback(df, text_column):
        df[text_column] = df[text_column].fillna('').astype(str)
        df[text_column] = df[text_column].apply(translate_to_english)
        df[text_column] = df[text_column].apply(lambda x: html.escape(str(x)))
        df['Sentiment'] = df[text_column].apply(analyze_sentiment_bert)
        return df
    #st.title("Feedback analyzer")


    #uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

    #if uploaded_file is not None:
        #try:
            #df = read_csv_safe(uploaded_file)

            #if df.empty:
            #  st.error("Uploaded CSV is empty or invalid.")
            # st.stop()

            #text_column = st.selectbox("Select the feedback text column", options=df.columns.tolist())

            #if st.button("Process Feedback"):
                #processed_df = process_feedback(df, text_column)
            # st.write("Processed feedback (first 10 rows):")
            # st.dataframe(processed_df[[text_column, 'Sentiment']].head(10))

        #except Exception as e:
        # st.error(f"Error: {e}")'''


    # -------------------- Dynamic Department Taxonomy Management --------------------
    def get_default_departments():
        return ['IT', 'HR', 'Finance', 'Customer Service', 'Technical', 'Payroll', 'Marketing', 'Operations']

    def load_departments():
        if 'departments' not in st.session_state:
            st.session_state['departments'] = get_default_departments()
        return st.session_state['departments']

    def save_departments(dept_list):
        st.session_state['departments'] = dept_list

    # -------------------- Embedded Department Classifier --------------------
    @st.cache_resource(show_spinner=False)
    def train_department_model_embedded(departments):
        # Use only feedback samples whose departments are in current taxonomy
        sample_data = {
                'Feedback': [
                'The IT team resolved my issue quickly',
                'HR was very helpful during onboarding',
                'Finance delayed the payment',
                'Customer service was excellent',
                'Technical support fixed the bug',
                'Payroll made a salary mistake',
                'Marketing launched a great campaign',
                'Operations handled logistics efficiently',
                'The IT support was slow to respond',
                'HR organized a fantastic training session',
                'Finance provided clear budget reports',
                'Customer service did not answer my calls',
                'Technical team upgraded our systems seamlessly',
                'Payroll processed bonuses correctly this time',
                'Marketing’s new strategy increased sales',
                'Operations ensured timely delivery of products',
                'IT infrastructure upgrades caused downtime',
                'HR resolved conflict between team members quickly',
                'Finance team was very transparent about expenses',
                'Customer service resolved my complaint professionally',
                'Technical department improved system security',
                'Payroll department handled tax deductions properly',
                'Marketing collaborated well with sales team',
                'Operations optimized warehouse management',
                'IT team provided excellent training on new software',
                'HR needs to improve communication',
                'Finance delayed month-end closing reports',
                'Customer service staff was very polite',
                'Technical support was unavailable during outage',
                'Payroll made accurate payments last quarter',
                'Marketing team exceeded expectations with social media',
                'Operations faced challenges during peak season',
                'IT helped automate routine tasks',
                'HR updated policies clearly and promptly',
                'Finance audit was thorough and helpful',
                'Customer service follow-up was lacking',
                'Technical experts solved critical bugs quickly',
                'Payroll system upgrade was smooth',
                'Marketing created engaging content',
                'Operations improved supply chain transparency',
                'App is crashing every time I try to open it',
                'I want to upgrade to premium version soon',
                'Payment was declined, not sure why',
                'Support team was friendly and solved my issue fast',
                'Logistics of delivery were smooth and well managed',
                'Recruiter reached out with helpful info about job openings',
                'Marketing email was very engaging and timely',
                'I had issues with my salary last month',
                'IT support helped reset my credentials quickly',
                'Operations team coordinated the move efficiently'
            ],
            'Department': [
                'IT', 'HR', 'Finance', 'Customer Service', 'Technical', 'Payroll', 'Marketing', 'Operations',
                'IT', 'HR', 'Finance', 'Customer Service', 'Technical', 'Payroll', 'Marketing', 'Operations',
                'IT', 'HR', 'Finance', 'Customer Service', 'Technical', 'Payroll', 'Marketing', 'Operations',
                'IT', 'HR', 'Finance', 'Customer Service', 'Technical', 'Payroll', 'Marketing', 'Operations',
                'IT', 'HR', 'Finance', 'Customer Service', 'Technical', 'Payroll', 'Marketing', 'Operations',
                'Technical', 'Sales', 'Finance', 'Customer Service', 'Operations', 'HR', 'Marketing', 'Payroll', 'IT', 'Operations'
            ]
        }
        df = pd.DataFrame(sample_data)

        # Filter to only departments currently managed
        df = df[df['Department'].isin(departments)]

        X = df['Feedback']
        y = df['Department']
        pipe = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('clf', LogisticRegression(max_iter=1000))
        ])
        pipe.fit(X, y)
        return pipe

    # -------------------- Visualization Helpers --------------------
    def generate_wordcloud(series):
        text = " ".join(series.fillna('').astype(str))
        return WordCloud(width=800, height=400, background_color='white').generate(text)

    # -------------------- Streamlit UI --------------------

    st.sidebar.header("⚙️ Department Taxonomy Management")

    # Load or initialize departments list
    departments = load_departments()

    # UI to add/remove departments dynamically
    with st.sidebar.expander("Manage Departments", expanded=True):
        st.write("Add or remove departments dynamically for classification.")

        # Show current departments with remove buttons
        to_remove = []
        for dept in departments:
            cols = st.columns([0.8, 0.2])
            cols[0].write(dept)
            if cols[1].button(f"Remove", key=f"remove_{dept}"):
                to_remove.append(dept)

        # Remove selected departments
        if to_remove:
            departments = [d for d in departments if d not in to_remove]
            save_departments(departments)
            st.rerun()

        # Add new department
        new_dept = st.text_input("Add New Department", key="new_dept_input")
        if st.button("Add Department"):
            new_dept_clean = new_dept.strip()
            if new_dept_clean and new_dept_clean not in departments:
                departments.append(new_dept_clean)
                save_departments(departments)
                st.rerun()
            else:
                st.warning("Department already exists or invalid input.")

    st.sidebar.markdown("---")
    st.sidebar.header("📤 Upload Files")

    general_file = st.sidebar.file_uploader("Upload General Feedback CSV", type=["csv"])

    # Compliance and Security Notes
    st.sidebar.markdown("""
    ---
    **Security & Compliance**  
    - Files are processed in-memory only, not saved to disk.  
    - Max file size: 10 MB.  
    - Downloaded CSVs can be optionally encrypted.  
    - Use secure HTTPS connection when deploying.  
    """)

    if general_file is not None:
        try:
            validate_file(general_file)
            general_df = read_csv_safe(general_file)

            if general_df.empty:
                st.error("General feedback CSV is empty or invalid.")
                st.stop()

            text_column = st.selectbox("Select the feedback text column", options=general_df.columns.tolist())

            # Process feedback (translation + sentiment)
            general_df = process_feedback(general_df, text_column)

            # Train department model dynamically based on current taxonomy
            department_model = train_department_model_embedded(departments)

            # -------------------- Keyword-based Department Classifier --------------------
            def keyword_department_classifier(text):
                text = text.lower()
                if any(word in text for word in [
                    'update', 'freeze', 'notification', 'bug', 'crash', 'error', 'issue', 'slow', 'lag', 'loading', 'hang', 'glitch', 'downtime',
                    'fix', 'patch', 'unavailable', 'ui', 'interface', 'navigation', 'settings', 'performance', 'feature'
                ]):
                    return 'Technical Support'
                if any(word in text for word in [
                    'payment', 'salary', 'invoice', 'finance', 'money', 'budget', 'pay', 'refund', 'billing', 'cost', 'price', 'charge',
                    'transaction', 'audit', 'expense', 'tax', 'payroll', 'compensation', 'wages', 'bonus', 'deduction', 'paycheck'
                ]):
                    return 'Finance'
                if any(word in text for word in [
                    'customer service', 'support', 'help', 'call', 'complaint', 'service', 'response', 'feedback', 'ticket', 'agent',
                    'chat', 'wait time', 'polite', 'friendly', 'slow response', 'responsive','fast reply'
                ]):
                    return 'Customer Service'
                if any(word in text for word in [
                    'hiring', 'onboarding', 'training', 'recruitment', 'hr', 'employee', 'policy',
                    'benefits', 'leave', 'promotion', 'team', 'manager', 'workplace', 'performance review'
                ]):
                    return 'HR'
                if any(word in text for word in [
                    'delivery', 'logistics', 'warehouse', 'shipment', 'operations', 'supply', 'inventory', 'shipping', 'tracking',
                    'order', 'fulfillment', 'scheduling', 'stock'
                ]):
                    return 'Operations'
                
                if any(word in text for word in [
                    'marketing', 'campaign', 'advertising', 'sales', 'brand', 'promotion', 'social media', 'strategy', 'content',
                    'engagement', 'launch', 'announcement'
                ]):
                    return 'Marketing'
                if any(word in text for word in [
                    'technical', 'system', 'security', 'infrastructure', 'upgrade', 'network', 'server', 'database', 'software',
                    'hardware', 'it', 'backend', 'configuration', 'deployment'
                ]):
                    return 'Technical'
                if any(word in text for word in [
                    'payroll', 'salary', 'bonus', 'deduction', 'tax', 'payment', 'paycheck', 'compensation', 'benefits', 'wages'
                ]):
                    return 'Payroll'
                if any(word in text for word in [
                    'great experience', 'everything works', 'looks fantastic', 'love this app', 'super easy to use',
                    'recommend', 'smooth', 'beautiful design', 'fantastic', 'nice ui'
                ]):
                    return 'Customer Experience'
                
                if any(word in text for word in ['upgrade', 'premium', 'subscribe', 'subscription', 'trial']):
                    return  'Sales'

                # Default fallback
                    return 'General'


            def combined_department_prediction(text):
                text = str(text)
                ml_probs = department_model.predict_proba([text])[0]
                ml_classes = department_model.classes_
                ml_max_index = np.argmax(ml_probs)
                ml_pred = ml_classes[ml_max_index]
                ml_conf = ml_probs[ml_max_index]

                keyword_pred = keyword_department_classifier(text)

                if ml_conf >= 0.6:
                    if keyword_pred == ml_pred or keyword_pred == "General":
                        return ml_pred
                    else:
                        return keyword_pred
                else:
                    if keyword_pred != "General":
                        return keyword_pred
                    else:
                        return "General"



    # WITH this combined approach:
            general_df['Department'] = general_df[text_column].apply(combined_department_prediction)


            # general_df['Department'] = general_df[text_column].apply(lambda x: department_model.predict([str(x)])[0])

            general_df['Sentiment'] = general_df['Sentiment'].fillna("UNKNOWN").str.upper().str.strip()
            general_df['Department'] = general_df['Department'].fillna("UNKNOWN").str.strip()

            selected_sentiment = st.selectbox("Filter by Sentiment", options=["All"] + sorted(general_df['Sentiment'].unique()))
            selected_department = st.selectbox("Filter by Department", options=["All"] + sorted(general_df['Department'].unique()))

            filtered_df = general_df.copy()
            if selected_sentiment != "All":
                filtered_df = filtered_df[filtered_df['Sentiment'] == selected_sentiment]
            if selected_department != "All":
                filtered_df = filtered_df[filtered_df['Department'] == selected_department]

            st.subheader("📊 Feedback Summary")
            show_filtered_metrics = st.checkbox("Show metrics based on filtered feedback only", value=True)

            target_df = filtered_df if show_filtered_metrics else general_df

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Feedbacks", len(target_df))
            col2.metric("👍 Positive", (target_df['Sentiment'] == 'POSITIVE').sum())
            col3.metric("👎 Negative", (target_df['Sentiment'] == 'NEGATIVE').sum())

            st.dataframe(filtered_df[[text_column, 'Sentiment', 'Department']], use_container_width=True)

            st.subheader("📈 Visualizations")
            

            st.plotly_chart(create_sentiment_pie_chart(filtered_df), use_container_width=True)


            dept_counts = filtered_df['Department'].value_counts().reset_index()
            dept_counts.columns = ['Department', 'Count']
            st.plotly_chart(px.bar(dept_counts, x='Department', y='Count', title='Feedback Volume by Department'), use_container_width=True)

            st.subheader("☁️ Word Cloud")
            wc = generate_wordcloud(filtered_df[text_column])
            fig_wc, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig_wc)

            st.subheader("📝 Sample Feedbacks")
            for sentiment in ['POSITIVE', 'NEGATIVE' ]:
                st.markdown(f"**{sentiment}**")
                samples = filtered_df[filtered_df['Sentiment'] == sentiment][text_column].head(3)
                for i, s in enumerate(samples):
                    st.write(f"{i+1}. {s}")

            st.subheader("📥 Download Filtered CSV")

            # Encrypt option for downloads
            encrypt_option = st.checkbox("Encrypt downloaded CSV files", value=False)

            def get_csv_bytes(df):
                return df.to_csv(index=False).encode('utf-8')

            if encrypt_option:
                filtered_csv = get_csv_bytes(filtered_df)
                encrypted_filtered_csv = encrypt_bytes(filtered_csv)
                st.download_button(
                    "📄 Download Encrypted Filtered Feedback CSV",
                    data=encrypted_filtered_csv,
                    file_name="filtered_feedback_encrypted.bin",
                    mime='application/octet-stream',
                    help="Encrypted file, decrypt with same app key."
                )

                all_csv = get_csv_bytes(general_df)
                encrypted_all_csv = encrypt_bytes(all_csv)
                st.download_button(
                    "📄 Download Encrypted All Feedback CSV",
                    data=encrypted_all_csv,
                    file_name="all_feedback_encrypted.bin",
                    mime='application/octet-stream',
                    help="Encrypted file, decrypt with same app key."
                )
            else:
                st.download_button(
                    "📄 Download Filtered Feedback CSV",
                    data=get_csv_bytes(filtered_df),
                    file_name="filtered_feedback.csv",
                    mime='text/csv'
                )
                st.download_button(
                    "📄 Download All Feedback CSV",
                    data=get_csv_bytes(general_df),
                    file_name="all_feedback.csv",
                    mime='text/csv'
                )

        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.text(traceback.format_exc())
    else:
        st.info("Please upload a General Feedback CSV file to start analysis.")



    end_time = time.time()  # End timing
    load_time = end_time - start_time

    st.sidebar.info(f"⏱️ UI Load Time: {load_time:.2f} seconds")


if __name__ == "__main__":
    main()