
-- Supabase / Postgres schema for Book Recommender demo
create table if not exists public.books (
    id bigint primary key,
    title text not null,
    author text,
    genre text,
    description text,
    created_at timestamptz default now()
);

create table if not exists public.users (
    id bigint primary key,
    name text not null,
    email text unique
);

create table if not exists public.ratings (
    id bigint primary key,
    user_id bigint not null references public.users(id) on delete cascade,
    book_id bigint not null references public.books(id) on delete cascade,
    rating int not null check (rating >= 1 and rating <= 5),
    created_at timestamptz default now()
);

create index if not exists idx_books_title on public.books using gin (to_tsvector('english', coalesce(title,'')));
create index if not exists idx_books_genre on public.books(genre);
create index if not exists idx_ratings_user on public.ratings(user_id);
create index if not exists idx_ratings_book on public.ratings(book_id);
