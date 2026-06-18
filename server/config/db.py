import os
import certifi
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "test_db")

if not MONGO_URI:
    raise RuntimeError("MONGO_URI environment variable is not set")

client=MongoClient(
    MONGO_URI,
    tls=True,
    tlsCAFile=certifi.where(),
    serverSelectionTimeoutMS=10000,
)
db=client[DB_NAME]

# User collection
users_collection = db["users"]
#  Document collection
chunk_collection=db["text"]
#  Chat collection
chat_history_collection=db["chat_history"]
#  Quiz collection
quizzes_collection=db["quizzes"]
quiz_history=db["history"]
