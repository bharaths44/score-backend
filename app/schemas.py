# schemas.py
from pydantic import BaseModel
from typing import List, Optional
import uuid


class UserSchema(BaseModel):
    id: uuid.UUID
    username: str


class NoteSchema(BaseModel):
    id: uuid.UUID
    title: str
    file_url: str
    text: Optional[str]
    category: Optional[str]
    tags: Optional[List[str]]
    created_at: Optional[str]


class CommentSchema(BaseModel):
    id: uuid.UUID
    note_id: uuid.UUID
    user_id: uuid.UUID
    content: str
    created_at: Optional[str]
