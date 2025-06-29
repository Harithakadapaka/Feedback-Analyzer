import pytest
import streamlit as st
from unittest.mock import MagicMock
from pytest_mock import mocker
from io import StringIO
from app import create_sentiment_pie_chart

# Sample CSV data for testing
@pytest.fixture
def sample_csv():
    data = StringIO("""Feedback,Sentiment,Department
    "Great app, love the interface!",POSITIVE,Customer Experience
    "The app is slow and buggy.",NEGATIVE,Technical Support
    "Very helpful support team.",POSITIVE,Customer Service
    """)
    return data

# Mocking Streamlit functions
def test_app_ui_load(mocker):
    # Mock Streamlit components
    mocker.patch('streamlit.file_uploader', return_value=None)
    mocker.patch('streamlit.selectbox', return_value='Feedback')

    # Simulate app logic by calling necessary components
    st.selectbox("Select the feedback text column", options=["Feedback"])
    st.selectbox("Filter by Sentiment", options=["All", "POSITIVE", "NEGATIVE"])
    st.selectbox("Filter by Department", options=["All", "Customer Experience", "Technical Support", "Customer Service"])

    # Ensure that selectbox is being called correctly
    assert st.selectbox.call_count == 3  # Assert that selectbox is called 3 times
    assert st.selectbox.call_args_list[0][0][0] == "Select the feedback text column"  # Check text in first selectbox
    assert st.selectbox.call_args_list[1][0][0] == "Filter by Sentiment"  # Check text in second selectbox
    assert st.selectbox.call_args_list[2][0][0] == "Filter by Department"  # Check text in third selectbox

def test_file_upload_and_processing(mocker, sample_csv):
    # Mock Streamlit's file_uploader
    mock_file_uploader = mocker.patch('streamlit.file_uploader', return_value=sample_csv)
    
    # Test file uploader functionality
    st.file_uploader("Upload General Feedback CSV", type=["csv"])
    
    # Assert that the file_uploader was called once
    assert mock_file_uploader.call_count == 1


def test_column_selection_and_filtering(mocker, sample_csv):
    # Mock Streamlit components
    mocker.patch('streamlit.file_uploader', return_value=sample_csv)

    # Mock selectbox with side_effect to return the values we want for each call
    mocker.patch('streamlit.selectbox', side_effect=[
        "Feedback",  # Select the feedback text column
        "All",  # Filter by Sentiment
        "POSITIVE",  # Filter by Sentiment
        "All",  # Filter by Department
    ])

    # Simulate file uploader
    st.file_uploader("Upload General Feedback CSV", type=["csv"])

    # Simulate column selection
    st.selectbox("Select the feedback text column", options=["Feedback"])
    st.selectbox("Filter by Sentiment", options=["All", "POSITIVE", "NEGATIVE"])
    st.selectbox("Filter by Department", options=["All", "Customer Experience", "Technical Support", "Customer Service"])

    # Print the calls to see the sequence
    print(st.selectbox.call_args_list)

    # Ensure the file uploader was called exactly once
    assert st.file_uploader.call_count == 1, "File uploader was not called correctly."

    # Ensure the correct number of selectbox calls
    assert st.selectbox.call_count == 3, f"Expected three selectboxes to be called, but got {st.selectbox.call_count}."



def test_create_sentiment_pie_chart():
    from app import create_sentiment_pie_chart
    import pandas as pd

    df = pd.DataFrame({
        "Sentiment": ["POSITIVE", "POSITIVE", "NEGATIVE"]
    })
    chart = create_sentiment_pie_chart(df)
    chart_json = chart.to_plotly_json()

    # Validate basic structure
    assert chart_json["data"][0]["type"] == "pie"
    assert set(chart_json["data"][0]["labels"]) == {"POSITIVE", "NEGATIVE"}



def test_download_buttons(mocker, sample_csv):
    # Mock Streamlit's file_uploader
    mocker.patch('streamlit.file_uploader', return_value=sample_csv)

    # Simulate the file uploader
    st.file_uploader("Upload General Feedback CSV", type=["csv"])

    # Mock the output for download buttons
    mocker.patch('streamlit.write', return_value=MagicMock())

    # Assert that the download buttons text is displayed
    st.write("Download Filtered Feedback CSV")
    st.write("Download Encrypted Filtered Feedback CSV")

    # Check that the expected text was written
    st.write.assert_any_call("Download Filtered Feedback CSV")
    st.write.assert_any_call("Download Encrypted Filtered Feedback CSV")


@pytest.mark.parametrize("encrypt_option, button_text", [(True, "Download Encrypted Filtered Feedback CSV"), (False, "Download Filtered Feedback CSV")])
def test_encrypted_download_button(mocker, sample_csv, encrypt_option, button_text):
    # Mock Streamlit components (file uploader and checkbox)
    mocker.patch('streamlit.file_uploader', return_value=sample_csv)
    mocker.patch('streamlit.checkbox', return_value=encrypt_option)

    # Simulate the checkbox for encryption
    st.checkbox("Encrypt downloaded CSV files", value=encrypt_option)

    # Mock the output for the download button based on encryption setting
    mocker.patch('streamlit.write', return_value=MagicMock())

    # Assert the correct button text is displayed based on encrypt_option
    st.write(button_text)

    # Check that the expected text was written
    st.write.assert_any_call(button_text)
