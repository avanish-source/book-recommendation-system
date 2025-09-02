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

def recommend_books(target_user, top_n=5):
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
            f"Recommended because {count_users} similar reader(s) gave it "
            f"an average of {avg_rating:.1f}★.\n"
            f"📖 Author: {book_info['author']} | 🎭 Genre: {book_info['genre']}\n"
            f"📝 Summary: {book_info['description']}"
        )
        
        recommendations.append((book_info["title"], reason, avg_rating))
    
    # Sort recommendations by average rating
    recommendations.sort(key=lambda x: x[2], reverse=True)
    
    return recommendations[:top_n]

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

if selected_user:
    st.subheader(f"📖 Your past ratings ({user_names[selected_user]}):")
    
    # Display past ratings in a styled markdown block
    # Note: Merging on 'user_id' from ratings and 'id' from books
    past = ratings[ratings["user_id"] == selected_user].merge(books, left_on="book_id", right_on="id")
    if not past.empty:
        for _, row in past.iterrows():
            st.markdown(
                f"""
                <div style="
                    background-color:#eef6ff;
                    padding:10px;
                    border-radius:10px;
                    margin-bottom:8px;">
                    <b>{row['title']}</b> — {row['rating']}★  
                    <span style="color:#666; font-size:13px;">
                        Author: {row['author']} | Genre: {row['genre']}
                    </span>
                    <p style="margin:4px 0; color:#444; font-size:14px;">
                        📝 Summary: {row['description']}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info("You haven't rated any books yet. Try rating some first!")

    # Display recommendations
    st.subheader(f"✨ Recommended for you:")
    with st.spinner("Generating recommendations..."):
        results = recommend_books(selected_user, top_n=5)
    
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
