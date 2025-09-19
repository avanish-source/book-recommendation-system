import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# --- Data Generation ---
# This section creates a dummy dataset of 100 properties and a simple user-rating dataset.
# The data now includes more realistic CRE attributes and placeholder images.

@st.cache_data
def load_data():
    """Loads and preprocesses dummy data for properties and user ratings."""
    data = {
        'id': range(1, 101),
        'name': [
            'Park Avenue Tower', 'Midtown Office Complex', 'Urban Retail Plaza',
            'Central Industrial Hub', 'Multi-Family Residence', 'Warehouse & Distribution',
            'Downtown Office Building', 'Suburban Strip Mall', 'Flex Space Warehouse',
            'Luxury High-Rise'
        ] * 10,
        'type': ['Office', 'Office', 'Retail', 'Industrial', 'Multi-family'] * 20,
        'subtype': [
            'Class A', 'Class B', 'Strip Mall', 'Warehouse', 'Condo',
            'Warehouse', 'Class B', 'Strip Mall', 'Flex Space', 'Class A'
        ] * 10,
        'location': ['Houston, TX', 'Austin, TX', 'Dallas, TX', 'San Antonio, TX', 'El Paso, TX'] * 20,
        'sqft': np.random.randint(5000, 150000, 100),
        'price_usd': np.random.randint(1_000_000, 100_000_000, 100),
        'cap_rate': np.round(np.random.uniform(3.5, 8.0, 100), 2),
        'occupancy_rate': np.round(np.random.uniform(0.75, 1.0, 100), 2),
        'image_url': [f"https://placehold.co/100x75/F1F5F9/265882?text=CRE-{i}" for i in range(1, 101)]
    }
    properties_df = pd.DataFrame(data)

    # Create dummy user ratings data for collaborative filtering
    # User 1 (The Office Investor) likes Class A office properties in Houston & Dallas.
    ratings_u1 = properties_df[
        (properties_df['type'] == 'Office') & 
        (properties_df['subtype'] == 'Class A') & 
        (properties_df['location'].isin(['Houston, TX', 'Dallas, TX']))
    ].head(5).assign(user_id=1, rating=1)

    # User 2 (The Retail Specialist) likes high cap rate retail properties.
    ratings_u2 = properties_df[
        (properties_df['type'] == 'Retail') & 
        (properties_df['cap_rate'] > 6.0)
    ].head(5).assign(user_id=2, rating=1)

    # User 3 (The New User) likes a few diverse properties.
    ratings_u3 = properties_df.sample(n=3).assign(user_id=3, rating=1)
    
    # Combine ratings
    ratings_df = pd.concat([ratings_u1, ratings_u2, ratings_u3])
    
    # Add some random ratings for other users to build a larger taste profile network
    for user_id in range(4, 21):
        random_properties = properties_df.sample(n=np.random.randint(2, 6))
        new_ratings = random_properties.assign(user_id=user_id, rating=1)
        ratings_df = pd.concat([ratings_df, new_ratings])

    return properties_df, ratings_df

# --- Recommendation Logic ---
# This section contains the core recommendation functions.

def content_based_recommendations(properties_df, property_id, num_recommendations=5):
    """
    Recommends properties based on feature similarity (content-based).
    The feature vector now includes more granular CRE data and is normalized.
    """
    # Create a feature matrix from numerical and categorical features
    numerical_features = properties_df[['sqft', 'price_usd', 'cap_rate', 'occupancy_rate']]
    
    # Normalize numerical features to give them equal weight
    scaler = MinMaxScaler()
    normalized_numerical = pd.DataFrame(scaler.fit_transform(numerical_features), columns=numerical_features.columns)

    # One-hot encode categorical features (type, subtype, and location)
    categorical_features = properties_df[['type', 'subtype', 'location']]
    encoded_features = pd.get_dummies(categorical_features, prefix=['type', 'subtype', 'loc'])

    # Combine all features into a single dataframe
    features_df = pd.concat([normalized_numerical, encoded_features], axis=1)
    
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
    user_item_matrix = ratings_df.pivot_table(index='user_id', columns='id', values='rating').fillna(0)
    
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
    
    st.write(f"The model found that users {similar_users.tolist()} have similar tastes to User {user_id}.")

    # Get the properties already liked by the current user
    current_user_liked_properties = set(user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index)
    
    # Recommend properties that similar users liked but the current user hasn't seen
    recommendation_ids = list(similar_users_liked_properties - current_user_liked_properties)
    
    # Return a DataFrame of the recommended properties
    recommended_df = properties_df[properties_df['id'].isin(recommendation_ids)]
    
    return recommended_df.head(num_recommendations)

# --- Streamlit UI ---
# This section builds the user interface.

st.set_page_c11onfig(layout="wide")

st.title('Commercial Real Estate Recommendation Engine')
st.markdown('### A Hybrid Approach to Data-Driven Property Discovery')
st.markdown(
    "Finding the right Commercial Real Estate property is a challenging and time-consuming process. "
    "This system uses a hybrid approach to quickly recommend properties that align with your "
    "investment strategy. It combines **Content-Based Filtering** (for finding similar properties) "
    "and **Collaborative Filtering** (for discovering properties based on the tastes of other investors)."
)

# Split the layout into two columns
col1, col2 = st.columns(2)

with col1:
    st.header('Content-Based Filtering')
    st.markdown(
        "**Find properties that are similar to a specific property.** This method analyzes key "
        "attributes like square footage, price, and location to find matches."
    )
    
    properties_df, ratings_df = load_data()
    property_options = properties_df[['id', 'name', 'location']]
    
    selected_property_name = st.selectbox(
        'Select a reference property:',
        options=property_options['name'],
        format_func=lambda x: f"{x} ({properties_df[properties_df['name'] == x]['location'].iloc[0]})"
    )
    
    if st.button('Get Content-Based Recommendations', use_container_width=True):
        selected_property_id = properties_df[properties_df['name'] == selected_property_name]['id'].iloc[0]
        st.subheader(f'Recommendations for "{selected_property_name}"')
        
        # Display the reference property details and image for context
        ref_property = properties_df[properties_df['id'] == selected_property_id].iloc[0]
        st.markdown("**Reference Property:**")
        
        # Use an expander for the reference property details
        with st.expander(ref_property['name'], expanded=True):
            st.image(ref_property['image_url'], use_column_width=True)
            st.write(
                f"**Type:** {ref_property['type']} | **Location:** {ref_property['location']} | **SQFT:** {ref_property['sqft']:,}"
            )
            st.write(f"**Price:** ${ref_property['price_usd']:,} | **Cap Rate:** {ref_property['cap_rate']:.2f}% | **Occupancy:** {ref_property['occupancy_rate']:.2f}%")
        
        st.markdown("---")
        
        recommendations = content_based_recommendations(properties_df, selected_property_id)
        st.dataframe(
            recommendations[['image_url', 'name', 'type', 'location', 'price_usd', 'cap_rate']],
            hide_index=True,
            use_container_width=True,
            column_config={
                "image_url": st.column_config.ImageColumn("Preview"),
                "price_usd": st.column_config.NumberColumn("Price (USD)", format="$%d"),
                "cap_rate": st.column_config.NumberColumn("Cap Rate (%)", format="%.2f"),
                "name": "Name",
                "type": "Type",
                "location": "Location"
            }
        )

with col2:
    st.header('Collaborative Filtering')
    st.markdown(
        "**Discover properties you might like based on other investors' tastes.** This method "
        "is great for finding opportunities you wouldn't have found through traditional search."
    )
    
    user_ids = sorted(ratings_df['user_id'].unique())
    selected_user_id = st.selectbox(
        'Select a user to get recommendations for:',
        user_ids
    )

    if st.button('Get Collaborative Filtering Recommendations', use_container_width=True):
        st.subheader(f'Recommendations for User {selected_user_id}')
        
        recommendations = collaborative_filtering_recommendations(properties_df, ratings_df, selected_user_id)
        
        if not recommendations.empty:
            st.dataframe(
                recommendations[['image_url', 'name', 'type', 'location', 'price_usd', 'cap_rate']],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "image_url": st.column_config.ImageColumn("Preview"),
                    "price_usd": st.column_config.NumberColumn("Price (USD)", format="$%d"),
                    "cap_rate": st.column_config.NumberColumn("Cap Rate (%)", format="%.2f"),
                    "name": "Name",
                    "type": "Type",
                    "location": "Location"
                }
            )
        else:
            st.info("No new recommendations found for this user. Try a different user or add more data!")

st.sidebar.header("How to Use this Demo")
st.sidebar.info(
    "**1. Content-Based:** Select a property and see similar ones appear on the left.\n\n"
    "**2. Collaborative Filtering:** Select a user ID and discover properties they'll love based on others' preferences.\n\n"
    "This demonstration is powered by a hybrid recommendation model trained on simulated CRE data. "
)
