import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import re

# --- Data Generation and Session State ---
# This section creates a dummy dataset and manages user interactions.

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
        'investment_type': ['Opportunistic', 'Core-Plus', 'Value-Add', 'Core', 'Opportunistic'] * 20,
        'year_built': np.random.randint(1950, 2020, 100),
        'units': np.random.randint(5, 500, 100),
        'advisor_name': ['Mitchell Belcher', 'Jeff Irish', 'Travis Prince', 'Chris Bruzas'] * 25
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

def get_user_interactions_df(ratings_df, selected_user_id):
    """Combines static ratings with dynamic session state interactions."""
    # Create a DataFrame from session state liked properties for the selected user
    user_likes = st.session_state.liked_properties.get(selected_user_id, [])
    user_views = st.session_state.viewed_properties.get(selected_user_id, [])
    
    # Combine liked and viewed properties into a single interaction set
    user_interactions = list(set(user_likes + user_views))
    
    if user_interactions:
        new_interactions_df = pd.DataFrame({
            'user_id': selected_user_id, 
            'id': user_interactions, 
            'rating': 1
        })
        # Merge with original ratings, dropping duplicates
        all_ratings_df = pd.concat([ratings_df, new_interactions_df], ignore_index=True).drop_duplicates(subset=['user_id', 'id'])
    else:
        all_ratings_df = ratings_df.copy()
        
    return all_ratings_df

def content_based_recommendations(properties_df, property_id, num_recommendations=5):
    """
    Recommends properties based on feature similarity (content-based).
    Returns a DataFrame of recommendations with explanations.
    """
    ref_property = properties_df[properties_df['id'] == property_id].iloc[0]
    
    # Create a feature matrix from numerical and categorical features
    numerical_features = properties_df[['sqft', 'price_usd', 'cap_rate', 'occupancy_rate', 'year_built', 'units']]
    
    # Normalize numerical features
    scaler = MinMaxScaler()
    normalized_numerical = pd.DataFrame(scaler.fit_transform(numerical_features), columns=numerical_features.columns)

    # One-hot encode categorical features
    categorical_features = properties_df[['type', 'subtype', 'location', 'investment_type']]
    encoded_features = pd.get_dummies(categorical_features, prefix=['type', 'subtype', 'loc', 'inv_type'])

    # Combine all features
    features_df = pd.concat([normalized_numerical, encoded_features], axis=1)
    
    # Calculate cosine similarity
    cosine_sim = cosine_similarity(features_df)
    
    idx = properties_df[properties_df['id'] == property_id].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in sim_scores[1:num_recommendations+1]]
    
    recommended_properties = properties_df.iloc[top_indices].copy()
    
    # Add a reason for each recommendation
    reasons = []
    for _, rec_row in recommended_properties.iterrows():
        reason = (f"Similar **type** ({rec_row['type']}), **location** ({rec_row['location']}), "
                  f"**year built** ({rec_row['year_built']}), and **number of units** ({rec_row['units']}).")
        reasons.append(reason)
    
    recommended_properties['reason'] = reasons
    
    return recommended_properties

def collaborative_filtering_recommendations(properties_df, ratings_df, user_id, user_personas, num_recommendations=5):
    """
    Recommends properties based on user-item ratings (collaborative filtering).
    Returns a DataFrame of recommendations and a detailed explanation.
    """
    all_ratings_df = get_user_interactions_df(ratings_df, user_id)
    user_item_matrix = all_ratings_df.pivot_table(index='user_id', columns='id', values='rating').fillna(0)
    
    if user_id not in user_item_matrix.index or len(user_item_matrix.index) < 2:
        return pd.DataFrame(), [], ""
        
    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)
    
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:3]
    
    similar_users_liked_properties = set()
    for sim_user_id in similar_users:
        liked_properties = user_item_matrix.loc[sim_user_id][user_item_matrix.loc[sim_user_id] > 0].index
        similar_users_liked_properties.update(liked_properties)
    
    current_user_liked_properties = set(user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index)
    
    common_properties = properties_df[properties_df['id'].isin(list(current_user_liked_properties & similar_users_liked_properties))]
    
    common_prop_list = ", ".join(list(set(common_properties['name'].tolist())))
    
    recommendation_ids = list(similar_users_liked_properties - current_user_liked_properties)
    recommended_df = properties_df[properties_df['id'].isin(recommendation_ids)]
    
    similar_users_names = [user_personas[uid]['name'] for uid in similar_users if uid in user_personas]
    
    explanation = (
        "This model finds properties based on the **shared taste** of investors like you. "
        "We first created a data matrix where each row represents an investor's liked and viewed properties. "
        "The system then used a mathematical technique called **cosine similarity** to measure how similar your 'like' history is to other investors. "
        f"The system found a strong taste match between you and investors like **{similar_users_names[0]}** and **{similar_users_names[1]}**. "
        f"You have a shared interest in properties like **{common_prop_list}**. Based on this shared taste, the model recommends other properties they liked but you haven't seen."
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
        .header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background-color: #003366;
            padding: 10px 20px;
            color: white;
        }
        .header-logo {
            font-weight: bold;
            font-size: 24px;
        }
        .header-nav a {
            color: white;
            text-decoration: none;
            margin-left: 20px;
        }
        .header-nav a:hover {
            text-decoration: underline;
        }
        /* Custom styling for property names on cards to match reference image */
        .card-title {
            color: #005C9E;
            font-weight: bold;
            font-size: 1.25rem;
            margin-bottom: 0.5rem;
        }
        .section-header {
            font-size: 1.5rem;
            font-weight: bold;
            color: #333333;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        .sidebar-card {
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding: 20px;
            background-color: #ffffff;
            position: sticky;
            top: 20px;
        }
    </style>
""", unsafe_allow_html=True)


# Initialize session state for navigation and interactions
if 'page' not in st.session_state:
    st.session_state.page = 'list'
if 'selected_property_id' not in st.session_state:
    st.session_state.selected_property_id = None
if 'liked_properties' not in st.session_state:
    st.session_state.liked_properties = {}
if 'viewed_properties' not in st.session_state:
    st.session_state.viewed_properties = {}
if 'current_user_id' not in st.session_state:
    st.session_state.current_user_id = 1

properties_df, ratings_df, user_personas = load_data()

# --- Dummy Header ---
st.markdown("""
<div class="header">
    <div style="display: flex; align-items: center;">
        <img src="https://assets.website-files.com/62c0199e1208a0d01d4a0a4c/62c0199e1208a032234a0a55_Berkadia_logowhite.svg" alt="Berkadia Logo" style="height: 30px; margin-right: 20px;">
    </div>
    <div class="header-nav">
        <a href="#">Services</a>
        <a href="#">Specialties</a>
        <a href="#">Properties</a>
        <a href="#">Insights</a>
        <a href="#">Research</a>
        <a href="#">About Us</a>
    </div>
</div>
<br>
""", unsafe_allow_html=True)

# Home Page: List of properties
if st.session_state.page == 'list':
    
    st.markdown("### Find your next investment opportunity.")

    # Persona selection on the main list page for personalized recommendations
    user_names = [user_personas[uid]['name'] for uid in sorted(user_personas.keys())]
    
    top_bar_cols = st.columns([0.7, 0.3])
    with top_bar_cols[1]:
        st.markdown("<p style='font-weight: bold; margin-bottom: 0;'>Logged In User</p>", unsafe_allow_html=True)
        selected_user_name = st.selectbox('User', user_names, label_visibility="collapsed", key='main_user_select')
        st.session_state.current_user_id = [uid for uid, info in user_personas.items() if info['name'] == selected_user_name][0]
    
    main_list_col, rec_sidebar_col = st.columns([3, 1])

    with main_list_col:
        search_query = st.text_input(
            'Enter a State, City, Zip code, or Property name',
            placeholder="e.g., Houston, TX, 77001, Park Avenue Tower"
        )
        
        st.markdown("---")
        st.markdown(f"**Properties** {len(properties_df)} results")
        
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
                    
                    # Title with Like button
                    title_col, like_col = st.columns([0.8, 0.2])
                    with title_col:
                        st.markdown(f"<h4 class='card-title'>{row['name']}</h4>", unsafe_allow_html=True)
                    with like_col:
                        is_liked = row['id'] in st.session_state.liked_properties.get(st.session_state.current_user_id, [])
                        heart_icon = '❤️' if is_liked else '🤍'
                        if st.button(heart_icon, key=f"like_{row['id']}"):
                            if not is_liked:
                                st.session_state.liked_properties.setdefault(st.session_state.current_user_id, []).append(row['id'])
                                st.rerun()
                            else:
                                st.session_state.liked_properties[st.session_state.current_user_id].remove(row['id'])
                                st.rerun()
    
                    st.markdown(f"<small>{row['address']}, {row['location']} {row['zip_code']}</small>", unsafe_allow_html=True)
                    st.markdown(f"**{row['investment_type']}**")
                    st.markdown(f"""
                        <p style='margin: 0; font-size: 14px;'>Built {row['year_built']} • Units {row['units']}</p>
                        <p style='margin: 0; font-size: 14px;'>{row['type']} Housing</p>
                        <p style='margin: 0; font-size: 14px;'>Advisor: <strong>{row['advisor_name']}</strong></p>
                    """, unsafe_allow_html=True)
                    
                    # Use a unique key for each button to avoid conflicts
                    if st.button('View Details', key=f"view_{row['id']}", use_container_width=True):
                        st.session_state.page = 'details'
                        st.session_state.selected_property_id = row['id']
                        st.session_state.viewed_properties.setdefault(st.session_state.current_user_id, []).append(row['id'])
                        st.rerun()

    with rec_sidebar_col:
        st.subheader('Suggested for you')
        st.markdown(f"""
            <div style="font-size: 14px; margin-top: -10px;">
                <p><strong>Persona:</strong> {user_personas[st.session_state.current_user_id]['persona']}</p>
                <small><i>Based on what other investors like you have viewed or liked.</i></small>
            </div>
        """, unsafe_allow_html=True)
        
        collab_recs_on_top, _, _ = collaborative_filtering_recommendations(properties_df, ratings_df, st.session_state.current_user_id, user_personas, num_recommendations=10)

        for rec_row in collab_recs_on_top.itertuples():
            with st.container(border=True):
                st.image(rec_row.image_url, use_container_width=True)
                st.markdown(f"**{rec_row.name}**")
                st.button('View Details', key=f"rec_sidebar_view_{rec_row.id}", use_container_width=True)


# Details Page: Single property listing and recommendations
elif st.session_state.page == 'details':
    if st.button("← Back to Listings"):
        st.session_state.page = 'list'
        st.rerun()

    selected_property_id = st.session_state.selected_property_id
    ref_property = properties_df[properties_df['id'] == selected_property_id].iloc[0]

    st.header(ref_property['name'])
    st.markdown(f"<small>{ref_property['address']}, {ref_property['location']} {ref_property['zip_code']}</small>", unsafe_allow_html=True)

    # Main content area
    main_col, sidebar_col = st.columns([3, 1])

    with main_col:
        st.image(ref_property['image_url'], use_container_width=True)

        st.markdown("<p class='section-header'>Property Overview</p>", unsafe_allow_html=True)
        with st.expander("Details", expanded=True):
            st.markdown(f"""
                - **Property Type:** {ref_property['type']}
                - **Property Subtype:** {ref_property['subtype']}
                - **Investment Type:** {ref_property['investment_type']}
                - **Square Footage:** {ref_property['sqft']:,} sqft
                - **Year Built:** {ref_property['year_built']}
                - **Units:** {ref_property['units']}
            """)

        with st.expander("Location & Demographics"):
            st.markdown(f"""
                - **Location:** {ref_property['location']}
                - **Zip Code:** {ref_property['zip_code']}
                - **Advisor:** {ref_property['advisor_name']}
                - *<small>Note: Additional demographic and location data would be displayed here in a full application.</small>*
            """, unsafe_allow_html=True)

        with st.expander("Financials"):
            st.markdown(f"""
                - **Cap Rate:** {ref_property['cap_rate']:.2f}%
                - **Occupancy Rate:** {ref_property['occupancy_rate']:.2f}%
                - *<small>Note: Detailed pro forma, revenue, and expense data would be displayed here.</small>*
            """, unsafe_allow_html=True)
    
    with sidebar_col:
        with st.container(border=True):
            st.subheader("Key Details")
            st.metric(label="Price", value=f"${(ref_property['price_usd'] / 1000000):.1f}M")
            st.metric(label="Cap Rate", value=f"{ref_property['cap_rate']:.2f}%")
            st.metric(label="Occupancy Rate", value=f"{ref_property['occupancy_rate']:.2f}%")
            
            # Like button with heart icon
            is_liked = selected_property_id in st.session_state.liked_properties.get(st.session_state.current_user_id, [])
            heart_icon = '❤️' if is_liked else '🤍'
            if st.button(f"{heart_icon} Like Property", key="like_button", use_container_width=True):
                if not is_liked:
                    st.session_state.liked_properties.setdefault(st.session_state.current_user_id, []).append(selected_property_id)
                    st.success("Property liked! This interaction will be used to improve recommendations.")
                else:
                    st.session_state.liked_properties[st.session_state.current_user_id].remove(selected_property_id)
                    st.info("Property unliked.")
                st.rerun()

            st.button("Contact Advisor", use_container_width=True)
    
    st.markdown("---")
    st.markdown('<div class="watermark">Powered by Prophecy AI</div>', unsafe_allow_html=True)
    
    # --- Recommendation Sections with unified view ---
    st.markdown("### Personalized Recommendations")
    rec_col1, rec_col2 = st.columns(2)
    
    with rec_col1:
        st.subheader('You May Also Like')
        st.markdown("<small><i>These properties are similar to the one you're viewing.</i></small>", unsafe_allow_html=True)
        content_recs = content_based_recommendations(properties_df, selected_property_id)
        
        # Display recommendations in a vertical list within the column
        for _, rec_row in content_recs.iterrows():
            with st.container(border=True):
                st.image(rec_row['image_url'], use_container_width=True)
                st.markdown(f"**{rec_row['name']}**")
                st.markdown(f"<small>{rec_row['location']}</small>", unsafe_allow_html=True)
                st.markdown(f"<small><i>{rec_row['reason']}</i></small>", unsafe_allow_html=True)

    with rec_col2:
        st.subheader('What Similar Investors Like')
        st.markdown("<small><i>These properties are recommended based on the tastes of investors like you.</i></small>", unsafe_allow_html=True)
        
        user_names = [user_personas[uid]['name'] for uid in sorted(user_personas.keys())]
        selected_user_name = st.selectbox(
            '**Simulate an investor profile:**',
            user_names
        )
        selected_user_id = [uid for uid, info in user_personas.items() if info['name'] == selected_user_name][0]
        st.session_state.current_user_id = selected_user_id

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
