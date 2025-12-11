from fastapi import FastAPI
from rag_engine import answer
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

class Query(BaseModel):
    pergunta: str

@app.post("/ask")
def ask(q: Query):
    resposta = answer(q.pergunta)
    return {"resposta": resposta}
