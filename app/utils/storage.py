from fastapi import UploadFile
from supabase import Client
import uuid
from ..config import get_db


def upload_to_supabase(file: UploadFile, db: Client = None) -> str:
    """
    Upload a file to Supabase storage and return the public URL

    Args:
        file (UploadFile): The file to upload
        db (Client): Supabase client instance

    Returns:
        str: Public URL of the uploaded file
    """
    if db is None:
        db = get_db()

    file_id = str(uuid.uuid4()) + "-" + file.filename
    file_data = file.file.read()

    # Upload to Supabase storage bucket named "notes"
    db.storage.from_("notes").upload(file_id, file_data)

    # Construct and return the public URL
    bucket_name = "notes"
    return f"{db.supabase_url}/storage/v1/object/public/{bucket_name}/{file_id}"
