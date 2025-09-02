import streamlit as st
import pandas as pd
from collections import defaultdict
import os

# Set page configuration for a wide layout
st.set_page_config(page_title="Book Recommendation System", layout="wide")

# Use st.cache_data to load dataframes only once for the entire app session.
# This significantly improves performance on Streamlit Cloud.
@st.cache_data
def load_data():
    """
    Loads all required data from CSV files.
    Includes error handling for missing files.
    """
    try:
        # The books dataframe uses 'id' instead of 'book_id'
        books_df = pd.read_csv("data/books.csv")
        # The users dataframe uses 'id' instead of 'user_id'
        users_df = pd.read_csv("data/users.csv")
        ratings_df = pd.read_csv("data/ratings.csv")
        return books_df, users_df, ratings_df
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e.filename}. Please ensure a 'data' folder exists in your repository "
                 "and contains books.csv, users.csv, and ratings.csv.")
        st.stop()

# Load data and build the user ratings profile
@st.cache_data
def build_user_profile(ratings_df):
    """
    Builds a dictionary of user ratings for faster lookups.
    """
    user_ratings = defaultdict(dict)
    for _, row in ratings_df.iterrows():
        # Using 'id' to reference books in the ratings profile
        user_ratings[row["user_id"]][row["book_id"]] = row["rating"]
    return user_ratings

# Load all data once at the start
books, users, ratings = load_data()
user_ratings = build_user_profile(ratings)

# ---------- Recommendation Logic ----------
def cosine_similarity(user1, user2):
    """Calculates cosine similarity between two users based on their ratings."""
    common_books = set(user_ratings[user1]).intersection(set(user_ratings[user2]))
    if not common_books:
        return 0
    
    # Calculate numerator (dot product)
    numerator = sum(user_ratings[user1][b] * user_ratings[user2][b] for b in common_books)
    
    # Calculate denominators (magnitude)
    den1 = sum(user_ratings[user1][b] ** 2 for b in common_books) ** 0.5
    den2 = sum(user_ratings[user2][b] ** 2 for b in common_books) ** 0.5
    
    return numerator / (den1 * den2) if den1 and den2 else 0

def recommend_books_collaborative(target_user, top_n=5):
    """
    Recommends books to a target user using collaborative filtering.
    """
    similarities = []
    # Find similarity with all other users
    for other_user in user_ratings:
        if other_user != target_user:
            sim = cosine_similarity(target_user, other_user)
            similarities.append((other_user, sim))
    
    # Sort by similarity in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Get top 5 similar users with a similarity score greater than 0
    top_similar_users = [u for u, s in similarities[:5] if s > 0]
    
    # Find candidate books from top similar users
    candidate_books = defaultdict(list)
    for sim_user in top_similar_users:
        for book_id, rating in user_ratings[sim_user].items():
            # Only consider books the target user hasn't read and which were highly rated
            if book_id not in user_ratings[target_user] and rating >= 4:
                candidate_books[book_id].append(rating)

    recommendations = []
    for book_id, ratings_list in candidate_books.items():
        # Calculate average rating for the candidate book
        avg_rating = sum(ratings_list) / len(ratings_list)
        count_users = len(ratings_list)
        
        # Get book details using the 'id' column from the books dataframe
        book_info = books[books["id"] == book_id].iloc[0]
        
        # Create a descriptive reason for the recommendation, including the description
        reason = (
            f"**Why this recommendation was made:**\n"
            f"Recommended because {count_users} similar reader(s) gave it "
            f"an average of {avg_rating:.1f}★.\n"
            f"📖 Author: {book_info['author']} | 🎭 Genre: {book_info['genre']}\n"
            f"📝 Summary: {book_info['description']}"
        )
        
        recommendations.append((book_info["title"], reason, avg_rating))
    
    # Sort recommendations by average rating
    recommendations.sort(key=lambda x: x[2], reverse=True)
    
    return recommendations[:top_n]


def recommend_books_content_based(target_user, top_n=5):
    """
    Recommends books to a target user using a content-based approach.
    """
    # Get the books the user has rated highly
    user_rated_books = ratings[ratings["user_id"] == target_user].merge(books, left_on="book_id", right_on="id")
    top_rated_books = user_rated_books[user_rated_books["rating"] >= 4]

    if top_rated_books.empty:
        return []

    # Get the top genres and authors from the user's highly-rated books
    favorite_genres = top_rated_books["genre"].tolist()
    favorite_authors = top_rated_books["author"].tolist()
    
    # Find candidate books that match these genres and authors
    recommendations_df = books[
        (books["genre"].isin(favorite_genres)) |
        (books["author"].isin(favorite_authors))
    ].copy()
    
    # Exclude books the user has already rated
    # CORRECTED: Use 'id_y' to reference the book ID from the merged dataframe
    rated_book_ids = user_rated_books["id_y"].tolist()
    recommendations_df = recommendations_df[~recommendations_df["id"].isin(rated_book_ids)]
    
    # For a simple approach, we'll sort by matching author/genre, but a more complex model could use tf-idf
    # For now, we'll just return a distinct list of up to top_n books.
    recommendations = []
    for _, row in recommendations_df.head(top_n).iterrows():
        reason = (
            f"**Why this recommendation was made:**\n"
            f"Recommended because it matches your interest in the "
            f"'{row['genre']}' genre and the author '{row['author']}'.\n"
            f"📝 Summary: {row['description']}"
        )
        recommendations.append((row["title"], reason, 0)) # Using 0 for rating as a placeholder
    
    return recommendations

# ---------- Streamlit UI ----------
st.title("📚 Personalized Book Recommendations")
st.write("Get smart suggestions based on what readers like you enjoyed!")

# Create a dictionary for easy mapping of user_id to name
# Note: The users dataframe now uses 'id'
user_names = users.set_index("id")["name"].to_dict()

# Use a selectbox to choose a user, with a custom format function to show names
selected_user = st.selectbox(
    "Choose a user:", 
    options=list(user_names.keys()), 
    format_func=lambda x: user_names[x]
)

# Add a radio button to select the recommendation method
recommendation_type = st.radio(
    "Choose recommendation type:",
    ('Collaborative Filtering', 'Content-Based Filtering'),
    horizontal=True
)

if selected_user:
    st.subheader(f"✨ Recommended for you:")
    with st.spinner("Generating recommendations..."):
        if recommendation_type == 'Collaborative Filtering':
            # Add the new description for collaborative filtering
            st.markdown(
                """
                Our **Collaborative Filtering** model works by finding other users with similar reading tastes
                to yours. We analyze how you and other readers have rated books, and then recommend
                titles that similar readers enjoyed but you haven't yet discovered.
                """
            )
            results = recommend_books_collaborative(selected_user, top_n=5)
        else:
            results = recommend_books_content_based(selected_user, top_n=5)
    
    if results:
        for title, reason, _ in results:
            st.markdown(
                f"""
                <div style="
                    background-color:#f9f9f9;
                    padding:15px;
                    border-radius:12px;
                    margin-bottom:12px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                    <h3 style="margin:0; color:#333;">{title}</h3>
                    <p style="margin:4px 0; color:#444; font-size:14px; white-space: pre-line;">
                        {reason}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.warning("No recommendations available yet. Try rating more books!")
