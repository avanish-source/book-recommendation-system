
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import Dataset, Reader, SVD

from supabase import create_client

# Supabase connection - expects env vars or you can set them here directly (not recommended)
SUPABASE_URL = os.environ.get('SUPABASE_URL') or 'https://YOUR_PROJECT.supabase.co'
SUPABASE_KEY = os.environ.get('SUPABASE_KEY') or 'YOUR_ANON_KEY'

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def fetch_table(name):
    rows = []
    start = 0
    page = 1000
    while True:
        resp = supabase.table(name).select('*').range(start, start+page-1).execute()
        data = resp.data or []
        rows.extend(data)
        if len(data) < page:
            break
        start += page
    return pd.DataFrame(rows)

def build_content_model(books_df):
    if books_df.empty:
        return None, None, None
    books_df = books_df.copy()
    books_df['text'] = books_df['genre'].fillna('') + ' ' + books_df['description'].fillna('')
    vectorizer = TfidfVectorizer(stop_words='english', max_features=50000, min_df=2)
    X = vectorizer.fit_transform(books_df['text'])
    return books_df, vectorizer, X

def get_content_recs(title_query, books_df, X, top_k=10):
    mask = books_df['title'].str.lower().str.contains(title_query.lower())
    if not mask.any():
        return pd.DataFrame()
    idx = books_df[mask].index[0]
    sims = linear_kernel(X[idx], X).flatten()
    top_idx = np.argsort(-sims)[1:top_k+1]
    recs = books_df.iloc[top_idx].copy()
    recs['score'] = sims[top_idx]
    return recs[['id','title','author','genre','description','score']]

def build_collab_model(ratings_df):
    if ratings_df.empty:
        return None
    reader = Reader(rating_scale=(1,5))
    data = Dataset.load_from_df(ratings_df[['user_id','book_id','rating']], reader)
    trainset = data.build_full_trainset()
    algo = SVD(n_factors=50, random_state=42)
    algo.fit(trainset)
    return algo

def get_collab_recs(algo, user_id, books_df, ratings_df, top_k=10):
    if algo is None:
        return pd.DataFrame()
    # find books user already rated
    rated = set(ratings_df[ratings_df['user_id']==user_id]['book_id'].astype(int).tolist())
    candidate_books = books_df[~books_df['id'].isin(rated)]['id'].tolist()
    preds = []
    for b in candidate_books:
        est = algo.predict(user_id, b).est
        preds.append((b, est))
    preds = sorted(preds, key=lambda x: x[1], reverse=True)[:top_k]
    recs = books_df[books_df['id'].isin([p[0] for p in preds])].copy()
    # attach predicted score
    score_map = {b: s for b,s in preds}
    recs['score'] = recs['id'].map(score_map)
    return recs[['id','title','author','genre','description','score']]
