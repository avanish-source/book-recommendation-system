import streamlit as st
import pandas as pd
from collections import defaultdict
 
# ---------- Load Data ----------
books = pd.read_csv("data/books.csv")       # book_id, title, author, genre
users = pd.read_csv("data/users.csv")       # user_id, name
ratings = pd.read_csv("data/ratings.csv")   # user_id, book_id, rating
 
# ---------- Build User Profiles ----------
user_ratings = defaultdict(dict)
for _, row in ratings.iterrows():
    user_ratings[row["user_id"]][row["book_id"]] = row["rating"]
 
# ---------- Similarity Function ----------
def cosine_similarity(user1, user2):
    common_books = set(user_ratings[user1]).intersection(set(user_ratings[user2]))
    if not common_books:
        return 0
    num = sum(user_ratings[user1][b] * user_ratings[user2][b] for b in common_books)
    den1 = sum(user_ratings[user1][b] ** 2 for b in common_books) ** 0.5
    den2 = sum(user_ratings[user2][b] ** 2 for b in common_books) ** 0.5
    return num / (den1 * den2) if den1 and den2 else 0
 
# ---------- Recommendation Logic ----------
def recommend_books(target_user, top_n=5):
    similarities = []
    for other_user in user_ratings:
        if other_user != target_user:
            sim = cosine_similarity(target_user, other_user)
            similarities.append((other_user, sim))
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_similar_users = [u for u, s in similarities[:5] if s > 0]
 
    candidate_books = defaultdict(list)
    for sim_user in top_similar_users:
        for book_id, rating in user_ratings[sim_user].items():
            if book_id not in user_ratings[target_user] and rating >= 4:
                candidate_books[book_id].append(rating)
 
    recommendations = []
    for book_id, ratings_list in candidate_books.items():
        avg_rating = sum(ratings_list) / len(ratings_list)
        count_users = len(ratings_list)
        book_info = books[books["book_id"] == book_id].iloc[0]
        reason = (
            f"Recommended because {count_users} similar reader(s) gave it "
            f"an average of {avg_rating:.1f}★.\n"
            f"📖 Author: {book_info['author']} | 🎭 Genre: {book_info['genre']}"
        )
        recommendations.append((book_info["title"], reason, avg_rating))
 
    recommendations.sort(key=lambda x: x[2], reverse=True)
    return recommendations[:top_n]
 
# ---------- Streamlit UI ----------
st.set_page_config(page_title="Book Recommendation System", layout="wide")
 
st.title("📚 Personalized Book Recommendations")
st.write("Get smart suggestions based on what readers like you enjoyed!")
 
# Select user
user_names = users.set_index("user_id")["name"].to_dict()
selected_user = st.selectbox("Choose a user:", options=list(user_names.keys()), format_func=lambda x: user_names[x])
 
if selected_user:
    st.subheader(f"📖 Your past ratings ({user_names[selected_user]}):")
 
    # Show past ratings
    past = ratings[ratings["user_id"] == selected_user].merge(books, on="book_id")
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
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info("You haven't rated any books yet. Try rating some first!")
 
    # Show recommendations
    results = recommend_books(selected_user, top_n=5)
 
    st.subheader(f"✨ Recommended for you:")
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