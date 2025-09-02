
import os
import streamlit as st
import pandas as pd
from recommender import fetch_table, build_content_model, get_content_recs, build_collab_model, get_collab_recs

st.set_page_config(page_title="Book Recommender Demo", layout="wide")
st.title("📚 Book Recommender Demo (Content + Collaborative)")

# Load small metadata preview from Supabase (or you can use local CSV by uncommenting)
use_supabase = True

if use_supabase:
    with st.spinner("Fetching data from Supabase..."):
        books_df = fetch_table('books')
        users_df = fetch_table('users')
        ratings_df = fetch_table('ratings')
else:
    # local fallback
    base = os.path.join(os.path.dirname(__file__), 'data')
    books_df = pd.read_csv(os.path.join(base, 'books_1000.csv'))
    users_df = pd.read_csv(os.path.join(base, 'users.csv'))
    ratings_df = pd.read_csv(os.path.join(base, 'ratings.csv'))

st.sidebar.header("Controls")
method = st.sidebar.radio("Method", ["Content-based", "Collaborative"])

books_df, vectorizer, X = build_content_model(books_df)
algo = build_collab_model(ratings_df)

if method == "Content-based":
    st.header("Find similar books by title")
    q = st.text_input("Type a title or partial title (e.g., 'Forgotten Forest')")
    if st.button("Recommend (content-based)"):
        if not q:
            st.warning("Type something first.")
        else:
            recs = get_content_recs(q, books_df, X, top_k=10)
            if recs.empty:
                st.info("No close match found. Try different words.")
            else:
                for _, row in recs.iterrows():
                    st.write(f\"**{row['title']}** — {row['author']}  \n{row['genre']}  \n{row['description']}  \nScore: {row['score']:.3f}\")
                    st.markdown('---')

else:
    st.header("Personalized recommendations (Collaborative)")
    user_list = users_df[['id','name']].drop_duplicates().set_index('id')['name'].to_dict()
    sel = st.selectbox("Pick a user", options=list(user_list.keys()), format_func=lambda x: user_list[x])
    if st.button("Recommend (collaborative)"):
        recs = get_collab_recs(algo, int(sel), books_df, ratings_df, top_k=10)
        if recs.empty:
            st.info("No recommendations available.")
        else:
            for _, row in recs.iterrows():
                st.write(f\"**{row['title']}** — {row['author']}  \n{row['genre']}  \n{row['description']}  \nPredicted Rating: {row['score']:.2f}\")
                st.markdown('---')
