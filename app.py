import streamlit as st
import requests
import json
import os
import pandas as pd
from azure.storage.blob import BlobServiceClient
import gc

# --- Configuration ---
# Récupérer l'URL de la fonction et la clé de fonction depuis les variables d'environnement
AZURE_FUNCTION_ENDPOINT = os.environ.get("AZURE_FUNCTION_ENDPOINT", "http://localhost:7071/api/recommend")
AZURE_FUNCTION_KEY = os.environ.get("AZURE_FUNCTION_KEY")
AZURE_STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")

# --- Helper Functions ---
@st.cache_data
def load_data_from_blob(connection_string, container_name, blob_name, is_pickle=False, is_json_lines=False):
    """
    Charge des données depuis un blob Azure.
    """
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        blob_data = blob_client.download_blob().readall()
        if is_pickle:
            return pickle.loads(blob_data)
        elif is_json_lines:
            data = []
            for line in blob_data.decode('utf-8').splitlines():
                data.append(json.loads(line))
            return data
        else:
            return json.loads(blob_data.decode('utf-8'))
    except Exception as e:
        st.error(f"An unexpected error occurred while loading {blob_name} from Blob Storage: {e}")
        return None

def optimize_dataframe_memory(df):
    """
    Optimise la mémoire utilisée par un DataFrame pandas.
    """
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        if df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    return df

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
    user_interactions_list = load_data_from_blob(AZURE_STORAGE_CONNECTION_STRING, "userinfosjson", "user_interactions.json", is_json_lines=True)
    user_interactions = pd.DataFrame(user_interactions_list)
    user_interactions = optimize_dataframe_memory(user_interactions)
    
    articles_metadata_list = load_data_from_blob(AZURE_STORAGE_CONNECTION_STRING, "processed-data", "articles_metadata.json", is_json_lines=True)
    articles_metadata = pd.DataFrame(articles_metadata_list)
    articles_metadata = optimize_dataframe_memory(articles_metadata)
    
    if user_interactions is not None:
        user_ids = sorted(list(set(user_interactions['user_id'])))
    else:
        user_ids = []
    
    if articles_metadata is not None:
        articles_metadata_dict = {str(article['article_id']): article for article in articles_metadata.to_dict('records')}
    else:
        articles_metadata_dict = {}
        
    # Libérer la mémoire
    del user_interactions_list
    del articles_metadata_list
    del user_interactions
    del articles_metadata
    gc.collect()
else:
    st.error("AZURE_STORAGE_CONNECTION_STRING environment variable not set. Cannot load data.")
    user_ids = []
    articles_metadata_dict = {}

if not user_ids:
    st.warning("No user IDs found or an error occurred loading them. Cannot proceed.")
    st.stop()

if not articles_metadata_dict:
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
                        article_info = articles_metadata_dict.get(str(rec_article_id)) # Ensure key is string if IDs are strings
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
