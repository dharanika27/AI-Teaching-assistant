import sys
from pathlib import Path

from fastapi import FastAPI


SERVER_DIR = Path(__file__).resolve().parent
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from auth.route import router as auth_router
from docs.route import router as doc_router
from chat.route import router as chat_router


app=FastAPI()



app.include_router(auth_router)
app.include_router(doc_router)
app.include_router(chat_router)


@app.get("/")
def home():
    return {"message":"Welcome to the User Management API"}
