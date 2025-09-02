
# Book Recommender Demo (Content-based + Collaborative)

This repo contains a demo-ready book recommender system with synthetic but realistic datasets.
It is designed for easy upload to Supabase and deployment on Streamlit Cloud.

## Files
- data/books_1000.csv - 1000 realistic book records
- data/users.csv - 100 realistic users
- data/ratings.csv - ~10,000 ratings
- supabase_setup.sql - SQL to create tables in Supabase
- load_data.py - Upload CSVs to Supabase (use with service role key)
- recommender.py - Recommender logic (connects to Supabase)
- app.py - Streamlit demo app (content + collaborative)
- requirements.txt - Python dependencies

## Quick start (Colab/local)
1. Upload the `data/` CSVs to your Supabase (or run load_data.py after setting env vars):
   - SUPABASE_URL=https://YOUR_PROJECT.supabase.co
   - SUPABASE_KEY=YOUR_SERVICE_ROLE_KEY
2. In Supabase SQL Editor run `supabase_setup.sql` to create tables.
3. Run `python load_data.py` to upsert data into Supabase.
4. To try locally, install requirements: `pip install -r requirements.txt`
5. Run the Streamlit app: `streamlit run app.py`
6. For deployment on Streamlit Cloud, push repo to GitHub and set SUPABASE_URL & SUPABASE_KEY as secrets in the app settings.

## Notes
- The dataset is synthetic and safe for demos.
- Use the service role key only in secure environments (not client-side).
