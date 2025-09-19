import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import re

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
        'address': [
            '123 Main St', '456 Oak Ave', '789 Pine Blvd', '101 Maple Dr', '202 Birch Rd'
        ] * 20,
        'zip_code': [77001, 78701, 75201, 78205, 79901] * 20,
        'sqft': np.random.randint(5000, 150000, 100),
        'price_usd': np.random.randint(1_000_000, 100_000_000, 100),
        'cap_rate': np.round(np.random.uniform(3.5, 8.0, 100), 2),
        'occupancy_rate': np.round(np.random.uniform(0.75, 1.0, 100), 2),
        'image_url': [f"https://placehold.co/100x75/F1F5F9/265882?text=CRE-{i}" for i in range(1, 101)],
        'investment_type': ['Opportunistic', 'Core-Plus', 'Value-Add', 'Core', 'Opportunistic'] * 20
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
    Returns a DataFrame of recommendations and a text-based explanation.
    """
    ref_property = properties_df[properties_df['id'] == property_id].iloc[0]
    
    # Create a feature matrix from numerical and categorical features
    numerical_features = properties_df[['sqft', 'price_usd', 'cap_rate', 'occupancy_rate']]
    
    # Normalize numerical features to give them equal weight
    scaler = MinMaxScaler()
    normalized_numerical = pd.DataFrame(scaler.fit_transform(numerical_features), columns=numerical_features.columns)

    # One-hot encode categorical features (type, subtype, and location)
    categorical_features = properties_df[['type', 'subtype', 'location', 'investment_type']]
    encoded_features = pd.get_dummies(categorical_features, prefix=['type', 'subtype', 'loc', 'inv_type'])

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
    
    # Generate the explanation text
    explanation = (
        f"These recommendations are based on a content-based model. We analyzed the selected property's attributes, "
        f"such as its **{ref_property['type']}** type, **{ref_property['subtype']}** subtype, and **{ref_property['location']}** location. "
        "The model then found other properties with the most similar characteristics, including shared financial metrics like **Cap Rate**."
    )
    
    return properties_df.iloc[top_indices], explanation

def collaborative_filtering_recommendations(properties_df, ratings_df, user_id, user_personas, num_recommendations=5):
    """
    Recommends properties based on user-item ratings (collaborative filtering).
    Returns a DataFrame of recommendations and a detailed explanation.
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
    
    # Find common properties liked by both the current user and similar users
    common_properties = properties_df[properties_df['id'].isin(list(current_user_liked_properties & similar_users_liked_properties))]
    common_prop_names = common_properties['name'].tolist()
    common_prop_list = ", ".join(common_prop_names)
    
    # Recommend properties that similar users liked but the current user hasn't seen
    recommendation_ids = list(similar_users_liked_properties - current_user_liked_properties)
    
    # Return a DataFrame of the recommended properties
    recommended_df = properties_df[properties_df['id'].isin(recommendation_ids)]
    
    # Generate the explanation text
    similar_users_names = [user_personas[uid]['name'] for uid in similar_users]
    explanation = (
        f"This model found a similarity between you and other investors, such as **{similar_users_names[0]}** and **{similar_users_names[1]}**. "
        f"You have a shared interest in properties like **{common_prop_list}**. "
        f"Based on this shared taste, the model recommends other properties that they liked but you haven't seen."
    )
    
    return recommended_df.head(num_recommendations), similar_users, explanation

# --- Streamlit UI ---
# This section builds the user interface.

st.set_page_config(layout="wide")

# Custom CSS for a professional, card-based layout
st.markdown("""
    <style>
        .stButton button {
            background-color: #0078D4;
            color: white;
            border-radius: 5px;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
        }
        .stButton button:hover {
            background-color: #005A9E;
        }
        .stContainer {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .main-listing-image {
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            width: 100%;
            height: auto;
        }
        .recommendation-card {
            min-width: 250px;
            max-width: 250px;
            border-radius: 8px;
            border: 1px solid #f0f0f0;
            padding: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            text-align: center;
        }
        .recommendation-card img {
            border-radius: 8px;
            width: 100%;
            height: auto;
            margin-bottom: 10px;
        }
        .watermark {
            position: fixed;
            bottom: 10px;
            right: 10px;
            font-size: 12px;
            color: #888888;
            background: #ffffff;
            padding: 5px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            z-index: 1000;
        }
    </style>
""", unsafe_allow_html=True)


# Initialize session state for navigation
if 'page' not in st.session_state:
    st.session_state.page = 'list'
if 'selected_property_id' not in st.session_state:
    st.session_state.selected_property_id = None

properties_df, ratings_df, user_personas = load_data()

# Home Page: List of properties
if st.session_state.page == 'list':
    st.title('Commercial Real Estate Recommendations')
    st.markdown("### Find your next investment opportunity.")
    
    search_query = st.text_input(
        'Enter a State, City, Zip code, or Property name',
        placeholder="e.g., Houston, TX, 77001, Park Avenue Tower"
    )

    st.markdown(f"**Properties** {len(properties_df)} results")
    st.markdown("---")
    
    # Filter properties based on search query
    if search_query:
        # Use a case-insensitive regex search across relevant columns
        filtered_df = properties_df[properties_df.apply(
            lambda row: any(re.search(search_query, str(row[col]), re.IGNORECASE) for col in ['name', 'location', 'zip_code', 'address']),
            axis=1
        )]
    else:
        filtered_df = properties_df.copy()

    # Display properties in a grid layout
    cols = st.columns(3)
    for i, row in filtered_df.iterrows():
        col = cols[i % 3]
        with col:
            with st.container(border=True):
                st.image(row['image_url'], use_container_width=True)
                st.subheader(row['name'])
                st.markdown(f"<small>{row['address']}, {row['location']} {row['zip_code']}</small>", unsafe_allow_html=True)
                st.markdown(f"**{row['investment_type']}**")
                if st.button('View Details', key=f"btn_{row['id']}", use_container_width=True):
                    st.session_state.page = 'details'
                    st.session_state.selected_property_id = row['id']
                    st.rerun()

# Details Page: Single property listing and recommendations
elif st.session_state.page == 'details':
    if st.button("← Back to Listings"):
        st.session_state.page = 'list'
        st.rerun()

    selected_property_id = st.session_state.selected_property_id
    ref_property = properties_df[properties_df['id'] == selected_property_id].iloc[0]

    # --- Main Property Listing View ---
    st.header(ref_property['name'])
    st.markdown(f"<small>{ref_property['address']}, {ref_property['location']} {ref_property['zip_code']}</small>", unsafe_allow_html=True)
    st.image(ref_property['image_url'], use_container_width=True)

    main_col1, main_col2, main_col3 = st.columns(3)
    with main_col1:
        st.metric(label="Price", value=f"${ref_property['price_usd']:,}")
    with main_col2:
        st.metric(label="Cap Rate", value=f"{ref_property['cap_rate']:.2f}%")
    with main_col3:
        st.metric(label="Occupancy Rate", value=f"{ref_property['occupancy_rate']:.2f}%")

    st.markdown("---")
    st.markdown('<div class="watermark">Powered by Prophecy AI</div>', unsafe_allow_html=True)

    # --- Recommendation Sections with unified view ---
    st.markdown("### Personalized Recommendations")
    rec_col1, rec_col2 = st.columns(2)

    with rec_col1:
        st.subheader('You May Also Like')
        st.markdown(
            "These properties are **similar** to the one you're viewing, based on key attributes."
        )
        content_recs, content_explanation = content_based_recommendations(properties_df, selected_property_id)
        st.info(content_explanation)
        
        # Display recommendations in a vertical list within the column
        for _, rec_row in content_recs.iterrows():
            with st.container(border=True):
                st.image(rec_row['image_url'], use_container_width=True)
                st.markdown(f"**{rec_row['name']}**")
                st.markdown(f"<small>{rec_row['location']}</small>", unsafe_allow_html=True)
                st.markdown(f"**{rec_row['investment_type']}**")

    with rec_col2:
        st.subheader('What Similar Investors Like')
        st.markdown(
            "These properties are recommended based on the tastes of investors like you."
        )
        
        user_names = [user_personas[uid]['name'] for uid in sorted(user_personas.keys())]
        selected_user_name = st.selectbox(
            '**Simulate an investor profile:**',
            user_names
        )
        selected_user_id = [uid for uid, info in user_personas.items() if info['name'] == selected_user_name][0]
        
        st.markdown(f"<p><strong>Persona:</strong> {user_personas[selected_user_id]['persona']}</p>", unsafe_allow_html=True)
        
        collaborative_recs, similar_users_ids, collab_explanation = collaborative_filtering_recommendations(properties_df, ratings_df, selected_user_id, user_personas)
        
        if not collaborative_recs.empty:
            st.info(collab_explanation)
            
            # Display recommendations in a vertical list within the column
            for _, rec_row in collaborative_recs.iterrows():
                with st.container(border=True):
                    st.image(rec_row['image_url'], use_container_width=True)
                    st.markdown(f"**{rec_row['name']}**")
                    st.markdown(f"<small>{rec_row['location']}</small>", unsafe_allow_html=True)
                    st.markdown(f"**{rec_row['investment_type']}**")
        else:
            st.info("No new recommendations found for this user. Try a different user or add more data!")

st.sidebar.header("How to Use this Demo")
st.sidebar.info(
    "**1. Search:** Start by using the search bar to find properties.\n\n"
    "**2. Listing Details:** Click 'View Details' on a card to see the full property listing.\n\n"
    "**3. Recommendations:** The two panels will show you personalized recommendations based on the hybrid model."
)
