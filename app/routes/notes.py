# notes.py
from fastapi import APIRouter, Depends, UploadFile, File

from supabase import Client

from ..config import get_db

from ..crud import create_note, get_all_notes
from ..utils.storage import upload_to_supabase
from ..utils.ai_utils import generate_tags_and_category

router = APIRouter()


@router.post("/upload_notes/")
async def upload_notes(
    files: list[UploadFile] = File(...), db: Client = Depends(get_db)
):
    notes_data = []
    for file in files:
        file_url = upload_to_supabase(file, db)

        # Basic text extraction based on file type
        text = ""
        content_type = file.content_type or ""

        if content_type.startswith("text/"):
            text = (await file.read()).decode("utf-8")
        # Add more file type handlers as needed

        ai_results = generate_tags_and_category(text)

        note_data = {
            "title": file.filename,
            "file_url": file_url,
            "text": text,
            "category": ai_results["category"]["name"],
            "tags": [tag["name"] for tag in ai_results["specific_tags"]],
        }

        note = create_note(db, note_data)
        notes_data.append(note)

    return {"notes": notes_data}


@router.get("/notes/")
def get_notes(db: Client = Depends(get_db)):
    return get_all_notes(db)
