import os
from supabase import create_client, Client


def get_db() -> Client:
    SUPABASE_URL: str = os.environ.get("SUPABASE_URL")
    SUPABASE_KEY: str = os.environ.get("SUPABASE_KEY")
    return create_client(SUPABASE_URL, SUPABASE_KEY)
