import os 
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def api_chave():
    return GOOGLE_API_KEY