# users.py
from fastapi import APIRouter
import uuid

router = APIRouter()

dummy_users = [
    {"id": str(uuid.uuid4()), "username": "Alice"},
    {"id": str(uuid.uuid4()), "username": "Bob"},
    {"id": str(uuid.uuid4()), "username": "Charlie"},
]


@router.get("/users/")
def get_dummy_users():
    return dummy_users
