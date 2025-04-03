# main.py
from fastapi import FastAPI
from .routes import notes, comments, users

app = FastAPI(title="AI-Powered Note Sharing Platform")

app.include_router(notes.router, prefix="/api")
app.include_router(comments.router, prefix="/api")
app.include_router(users.router, prefix="/api")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
