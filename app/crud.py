from supabase import Client


def create_note(db: Client, note_data: dict):
    """Create a new note in Supabase."""
    result = db.table("notes").insert(note_data).execute()
    return result.data[0] if result.data else None


def get_all_notes(db: Client):
    """Get all notes from Supabase."""
    result = db.table("notes").select("*").execute()
    return result.data


def add_comment(db: Client, comment_data: dict):
    """Add a comment to a note in Supabase."""
    result = db.table("comments").insert(comment_data).execute()
    return result.data[0] if result.data else None


def get_comments(db: Client, note_id: str):
    """Get all comments for a specific note."""
    result = db.table("comments").select("*").eq("note_id", note_id).execute()
    return result.data
