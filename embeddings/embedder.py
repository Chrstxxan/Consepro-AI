import os
import openai
from dotenv import load_dotenv
load_dotenv()


openai.api_key = os.getenv("OPENAI_API_KEY")

MODEL_EMBED = "text-embedding-3-large"

def embed(text: str):
    resp = openai.embeddings.create(
        model=MODEL_EMBED,
        input=text
    )
    return resp.data[0].embedding
