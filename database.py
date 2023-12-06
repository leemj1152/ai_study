from firebase_admin import credentials
from firebase_admin import db
from dotenv import load_dotenv
import os


def database():
    load_dotenv()
    firebase_key = os.getenv("FIREBASE_KEY")
    cred = credentials.Certificate(firebase_key)
    print(firebase_key)
