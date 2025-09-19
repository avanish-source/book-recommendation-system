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

    # Define user personas with their names and descriptions
    user_personas = {
        1: {'name': 'John S.', 'persona': 'The "Office Investor" with a focus on high-end, stable assets.'},
        2: {'name': 'Emily W.', 'persona': 'The "Retail Specialist" who targets high-yield retail properties.'},
        3: {'name': 'Chris L.', 'persona': 'The "New Investor" who is exploring diverse property types.'},
        4: {'name': 'Sarah T.', 'persona': 'The "Diversified Investor" with broad interests.'},
        5: {'name': 'David R.', 'persona': 'The "Suburban Expert" who prefers properties outside the city center.'},
        6: {'name': 'Jessica B.', 'persona': 'The "Industrial Pro" with a history of liking warehouse and distribution centers.'},
        7: {'name': 'Michael D.', 'persona': 'The "Value-Add Hunter" who looks for undervalued properties with low occupancy rates.'},
        8: {'name': 'Olivia P.', 'persona': 'The "Multi-Family Maven" specializing in residential complexes.'},
        9: {'name': 'Daniel A.', 'persona': 'The "Texas Native" with a portfolio concentrated in Texas markets.'},
        10: {'name': 'Sophia M.', 'persona': 'The "Cap Rate Chaser" who prioritizes investment yield above all else.'},
        # Adding more for demonstration purposes
        11: {'name': 'Liam J.', 'persona': 'The "Urban Core Enthusiast" focusing on downtown locations.'},
        12: {'name': 'Ava K.', 'persona': 'The "Growth Seeker" who invests in high-growth areas.'},
        13: {'name': 'Noah V.', 'persona': 'The "Conservative Investor" who avoids high-risk properties.'},
        14: {'name': 'Isabella G.', 'persona': 'The "Flexible Space Flipper" who prefers versatile commercial spaces.'},
        15: {'name': 'Mason C.', 'persona': 'The "Hospitality Mogul" with a liking for hotel properties.'},
        16: {'name': 'Chloe L.', 'persona': 'The "Small Business Supporter" who invests in small commercial properties.'},
        17: {'name': 'Ethan H.', 'persona': 'The "Global Player" with a wide range of interests.'},
        18: {'name': 'Mia W.', 'persona': 'The "Tech Hub Investor" focusing on properties near tech companies.'},
        19: {'name': 'Alexander T.', 'persona': 'The "Steady Hand" who prefers long-term, stable investments.'},
        20: {'name': 'Harper R.', 'persona': 'The "E-commerce Investor" who buys industrial properties for logistical needs.'},
    }

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

    return properties_df, ratings_df, user_personas

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
    
    # Get the properties already liked by the current user
    current_user_liked_properties = set(user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index)
    
    # Recommend properties that similar users liked but the current user hasn't seen
    recommendation_ids = list(similar_users_liked_properties - current_user_liked_properties)
    
    # Return a DataFrame of the recommended properties
    recommended_df = properties_df[properties_df['id'].isin(recommendation_ids)]
    
    return recommended_df.head(num_recommendations), similar_users

# --- Streamlit UI ---
# This section builds the user interface.

st.set_page_config(layout="wide")

st.title('Commercial Real Estate Recommendation Engine')
st.markdown("### Powered by a Hybrid Recommendation Model")
st.markdown("---")

properties_df, ratings_df, user_personas = load_data()

# Main Property Selection UI
selected_property_name = st.selectbox(
    '**Select a Property to View its Listing and Recommendations:**',
    options=properties_df['name'],
    format_func=lambda x: f"{x} ({properties_df[properties_df['name'] == x]['location'].iloc[0]})"
)

selected_property_id = properties_df[properties_df['name'] == selected_property_name]['id'].iloc[0]
ref_property = properties_df[properties_df['id'] == selected_property_id].iloc[0]

# --- Main Property Listing View ---
st.header(ref_property['name'])
st.image(ref_property['image_url'], use_column_width=True)

main_col1, main_col2, main_col3 = st.columns(3)
with main_col1:
    st.metric(label="Price", value=f"${ref_property['price_usd']:,}")
with main_col2:
    st.metric(label="Cap Rate", value=f"{ref_property['cap_rate']:.2f}%")
with main_col3:
    st.metric(label="Occupancy Rate", value=f"{ref_property['occupancy_rate']:.2f}%")

st.markdown("### Property Details")
st.markdown(f"**Type:** {ref_property['type']} | **Subtype:** {ref_property['subtype']} | **Location:** {ref_property['location']} | **SQFT:** {ref_property['sqft']:,}")

st.markdown("---")

# --- Recommendation Sections ---
col1, col2 = st.columns(2)

# Content-Based Filtering Section (left column)
with col1:
    st.subheader('Properties You May Also Like')
    st.markdown(
        "These are properties **similar** to the one you're viewing, based on key attributes like "
        "type, location, and financial metrics."
    )
    
    recommendations = content_based_recommendations(properties_df, selected_property_id)
    
    st.dataframe(
        recommendations[['image_url', 'name', 'type', 'location']],
        hide_index=True,
        use_container_width=True,
        column_config={
            "image_url": st.column_config.ImageColumn("Preview"),
            "name": "Name",
            "type": "Type",
            "location": "Location"
        }
    )

# Collaborative Filtering Section (right column)
with col2:
    st.subheader('Properties Similar Investors Like')
    st.markdown(
        "Discover properties you might not have found on your own. This model recommends properties "
        "that other investors with a similar taste profile have liked."
    )
    
    user_names = [user_personas[uid]['name'] for uid in sorted(user_personas.keys())]
    selected_user_name = st.selectbox(
        '**Simulate an investor profile:**',
        user_names
    )
    selected_user_id = [uid for uid, info in user_personas.items() if info['name'] == selected_user_name][0]
    
    st.markdown(f"**User Persona:** {user_personas[selected_user_id]['persona']}")
    
    recommendations, similar_users_ids = collaborative_filtering_recommendations(properties_df, ratings_df, selected_user_id)
    similar_users_names = [user_personas[uid]['name'] for uid in similar_users_ids]

    if not recommendations.empty:
        st.info(
            f"These recommendations are based on properties liked by investors with similar tastes, "
            f"such as **{similar_users_names[0]}** and **{similar_users_names[1]}**."
        )
        st.dataframe(
            recommendations[['image_url', 'name', 'type', 'location']],
            hide_index=True,
            use_container_width=True,
            column_config={
                "image_url": st.column_config.ImageColumn("Preview"),
                "name": "Name",
                "type": "Type",
                "location": "Location"
            }
        )
    else:
        st.info("No new recommendations found for this user. Try a different user or add more data!")

st.sidebar.header("How to Use this Demo")
st.sidebar.info(
    "**1. Main Listing:** Select a property to see its full details.\n\n"
    "**2. You May Also Like:** The left panel shows recommendations based on the features of the property you're viewing.\n\n"
    "**3. Similar Investors Like:** The right panel shows recommendations tailored to a user persona, demonstrating the power of collaborative filtering."
)
