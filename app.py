import streamlit as st
import requests
import json
import os
import pandas as pd
from azure.storage.blob import BlobServiceClient

# --- Configuration ---
# Récupérer l'URL de la fonction et la clé de fonction depuis les variables d'environnement
AZURE_FUNCTION_ENDPOINT = os.environ.get("AZURE_FUNCTION_ENDPOINT", "http://localhost:7071/api/recommend")
AZURE_FUNCTION_KEY = os.environ.get("AZURE_FUNCTION_KEY")
AZURE_STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")

ARTICLES_METADATA_PATH = "processed_data/articles_metadata.json"

# --- Helper Functions ---
@st.cache_data
def load_user_ids_from_blob(connection_string, container_name="userinfosjson", blob_name="user_interactions.json"):
    """Loads user IDs from the user interactions JSONL file in Blob Storage."""
    user_ids = set()
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        blob_data = blob_client.download_blob().readall()
        user_interactions = json.loads(blob_data.decode('utf-8'))
        for item in user_interactions:
            if 'user_id' in item:
                user_ids.add(item['user_id'])
        return sorted(list(user_ids))
    except Exception as e:
        st.error(f"An unexpected error occurred while loading user IDs from Blob Storage: {e}")
        return []

@st.cache_data
def load_articles_metadata(file_path):
    """Loads articles metadata from the JSONL file."""
    articles_metadata = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    article = json.loads(line)
                    if 'article_id' in article:
                        articles_metadata[str(article['article_id'])] = article # Ensure key is string
                except json.JSONDecodeError:
                    st.warning(f"Skipping malformed JSON line in {file_path}: {line[:100]}...")
                    continue
        return articles_metadata
    except FileNotFoundError:
        st.error(f"Error: Articles metadata file not found at {file_path}")
        return {}
    except Exception as e:
        st.error(f"An unexpected error occurred while loading articles metadata: {e}")
        return {}

def get_recommendations(user_id, n_recommendations=5):
    """Calls the Azure Function to get recommendations."""
    headers = {
        "Content-Type": "application/json",
        "x-functions-key": AZURE_FUNCTION_KEY
    }
    payload = {"user_id": user_id, "n_recommendations": n_recommendations}
    try:
        response = requests.post(AZURE_FUNCTION_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("Connection Error: Could not connect to the Azure Function. Please ensure it is running locally.")
        return None
    except requests.exceptions.Timeout:
        st.error("Timeout Error: The request to the Azure Function timed out.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP Error: {e.response.status_code} - {e.response.text}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"An unexpected error occurred during the request: {e}")
        return None
    except json.JSONDecodeError:
        st.error("Error: Could not decode JSON response from the Azure Function.")
        return None

# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="Article Recommender")

st.title("Article Recommender System")
st.markdown("---")

# Load data
if AZURE_STORAGE_CONNECTION_STRING:
    user_ids = load_user_ids_from_blob(AZURE_STORAGE_CONNECTION_STRING)
else:
    st.error("AZURE_STORAGE_CONNECTION_STRING environment variable not set. Cannot load user IDs.")
    user_ids = []

articles_metadata = load_articles_metadata(ARTICLES_METADATA_PATH)

if not user_ids:
    st.warning("No user IDs found or an error occurred loading them. Cannot proceed.")
    st.stop()

if not articles_metadata:
    st.warning("No article metadata found or an error occurred loading it. Recommendations will not show full details.")

# User selection
st.header("Get Recommendations")
selected_user_id = st.selectbox("Select a User ID", user_ids)

if st.button("Get 5 Recommendations"):
    if selected_user_id:
        with st.spinner(f"Getting recommendations for user {selected_user_id}..."):
            recommendations = get_recommendations(selected_user_id, n_recommendations=5)

        if recommendations:
            st.success("Recommendations received!")
            st.subheader(f"Recommendations for User ID: {selected_user_id}")

            if isinstance(recommendations, list) and recommendations:
                # Create columns for a card-like display
                cols = st.columns(3) # Adjust number of columns as needed

                for i, rec_article_id in enumerate(recommendations):
                    with cols[i % 3]: # Distribute cards across columns
                        article_info = articles_metadata.get(str(rec_article_id)) # Ensure key is string if IDs are strings
                        if article_info:
                            st.markdown(f"**Title:** {article_info.get('title', 'N/A')}")
                            st.markdown(f"**Category:** {article_info.get('category', 'N/A')}")
                            st.markdown(f"**URL:** [Link]({article_info.get('url', '#')})")
                            st.markdown("---")
                        else:
                            st.warning(f"Details for article ID {rec_article_id} not found.")
                            st.markdown(f"**Article ID:** {rec_article_id}")
                            st.markdown("---")
            else:
                st.info("No recommendations returned for this user.")
        else:
            st.error("Failed to retrieve recommendations. Check the error messages above.")
    else:
        st.warning("Please select a User ID.")

st.markdown("---")
st.info("Ensure your Azure Function is running locally on port 7071 for this application to work.")
