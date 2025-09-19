import streamlit as st
import pandas as pd
import numpy as np # Import numpy
from sklearn.metrics.pairwise import cosine_similarity

# --- Data Generation ---
# This section creates a dummy dataset of 100 properties and a simple user-rating dataset.
# In a real-world scenario, you would load this data from a database or CSV file.

@st.cache_data
def load_data():
    """Loads and preprocesses dummy data for properties and user ratings."""
    properties_df = pd.DataFrame({
        'id': range(1, 101),
        'name': [f'Property {i}' for i in range(1, 101)],
        'type': ['Apartment', 'House', 'Apartment', 'Condo', 'House'] * 20,
        'location': ['City A', 'City B', 'City C', 'City D', 'City E'] * 20,
        'bedrooms': [2, 3, 1, 2, 4, 3, 2, 1, 3, 4] * 10,
        'sqft': [1200, 1800, 800, 1500, 2500] * 20,
        'price_usd': [250000, 450000, 150000, 300000, 600000] * 20,
    })

    # Create dummy user ratings data for collaborative filtering
    # A value of 1 means the user "liked" the property
    ratings = []
    # User 1 likes properties 1, 5, 10, 15
    ratings.extend([{'user_id': 1, 'property_id': i, 'rating': 1} for i in [1, 5, 10, 15]])
    # User 2 likes properties 2, 6, 10, 16
    ratings.extend([{'user_id': 2, 'property_id': i, 'rating': 1} for i in [2, 6, 10, 16]])
    # User 3 likes properties 1, 2, 3, 4, 5
    ratings.extend([{'user_id': 3, 'property_id': i, 'rating': 1} for i in [1, 2, 3, 4, 5]])

    # Add some random ratings to simulate a larger dataset
    for user_id in range(4, 21):
        for _ in range(5):
            ratings.append({'user_id': user_id, 'property_id': np.random.randint(1, 101), 'rating': 1}) # Corrected line

    ratings_df = pd.DataFrame(ratings)

    return properties_df, ratings_df

# --- Recommendation Logic ---
# This section contains the core recommendation functions.

def content_based_recommendations(properties_df, property_id, num_recommendations=5):
    """
    Recommends properties based on feature similarity (content-based).
    It uses a simplified feature vector for demonstration.
    """
    # Create a feature matrix from numerical and categorical features
    features_df = properties_df[['bedrooms', 'sqft', 'price_usd']]
    # One-hot encode categorical features (type and location)
    encoded_features = pd.get_dummies(properties_df[['type', 'location']], prefix=['type', 'loc'])
    features_df = pd.concat([features_df, encoded_features], axis=1)

    # Calculate the cosine similarity between properties
    cosine_sim = cosine_similarity(features_df)

    # Find the index of the selected property
    idx = properties_df[properties_df['id'] == property_id].index[0]

    # Get a list of similarity scores for the selected property
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the properties based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the top N most similar properties (excluding the property itself)
    top_indices = [i[0] for i in sim_scores[1:num_recommendations+1]]

    return properties_df.iloc[top_indices]

def collaborative_filtering_recommendations(properties_df, ratings_df, user_id, num_recommendations=5):
    """
    Recommends properties based on user-item ratings (collaborative filtering).
    This is a simplified user-based collaborative filtering approach.
    """
    # Create a user-item matrix
    user_item_matrix = ratings_df.pivot_table(index='user_id', columns='property_id', values='rating').fillna(0)

    # Calculate user similarity (cosine similarity)
    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

    # Find the most similar users to the current user
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:3] # Top 2 similar users

    # Get the properties liked by similar users
    similar_users_liked_properties = set()
    for sim_user_id in similar_users:
        liked_properties = user_item_matrix.loc[sim_user_id][user_item_matrix.loc[sim_user_id] > 0].index
        similar_users_liked_properties.update(liked_properties)

    # Get the properties already liked by the current user
    current_user_liked_properties = set(user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index)

    # Recommend properties that similar users liked but the current user hasn't seen
    recommendation_ids = list(similar_users_liked_properties - current_user_liked_properties)

    # Return a DataFrame of the recommended properties
    recommended_df = properties_df[properties_df['id'].isin(recommendation_ids)]

    return recommended_df.head(num_recommendations)

# --- Streamlit UI ---
# This section builds the user interface.

st.set_page_config(layout="wide")

st.title('Property Recommendation Engine')
st.markdown('Explore property recommendations using two different methods: **Content-Based** and **Collaborative Filtering**.')

# Load data and get lists for selectboxes
properties_df, ratings_df = load_data()
property_names = properties_df['name'].tolist()
user_ids = sorted(ratings_df['user_id'].unique())

# Split the layout into two columns
col1, col2 = st.columns(2)

with col1:
    st.header('Content-Based Filtering')
    st.markdown('Recommends properties similar to one you select based on their features.')

    selected_property_name = st.selectbox(
        'Select a reference property:',
        property_names
    )

    if st.button('Get Content-Based Recommendations', use_container_width=True):
        selected_property_id = properties_df[properties_df['name'] == selected_property_name]['id'].iloc[0]
        st.subheader(f'Recommendations for {selected_property_name}')
        st.dataframe(
            content_based_recommendations(properties_df, selected_property_id),
            hide_index=True,
            use_container_width=True
        )

with col2:
    st.header('Collaborative Filtering')
    st.markdown('Recommends properties based on what other users with similar tastes liked.')

    selected_user_id = st.selectbox(
        'Select a user to get recommendations for:',
        user_ids
    )

    if st.button('Get Collaborative Filtering Recommendations', use_container_width=True):
        st.subheader(f'Recommendations for User {selected_user_id}')
        st.dataframe(
            collaborative_filtering_recommendations(properties_df, ratings_df, selected_user_id),
            hide_index=True,
            use_container_width=True
        )

st.sidebar.header("About this Demo")
st.sidebar.info(
    "This application demonstrates two types of recommendation systems:\n\n"
    "**1. Content-Based:** Recommends items based on the features of items the user liked.\n"
    "**2. Collaborative Filtering:** Recommends items based on the preferences of similar users.\n\n"
    "This is a simplified example using dummy data. In a real-world system, you would use a much larger, "
    "more diverse dataset and more advanced algorithms."
)
st.sidebar.image("https://placehold.co/300x200/F1F5F9/265882?text=Streamlit+UI")