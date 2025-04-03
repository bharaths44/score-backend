from fastapi import APIRouter, Depends

from supabase import Client

from ..config import get_db

from ..crud import add_comment, get_comments
from ..schemas import CommentSchema


router = APIRouter()


@router.post("/add_comment/")
async def add_comment_api(comment: CommentSchema, db: Client = Depends(get_db)):
    return add_comment(db, comment.dict())


@router.get("/comments/{note_id}")
async def get_comments_api(note_id: str, db: Client = Depends(get_db)):
    return get_comments(db, note_id)
