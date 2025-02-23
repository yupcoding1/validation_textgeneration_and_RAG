from fastapi import FastAPI
from pydantic import BaseModel
from services import validate

app = FastAPI()

class TextRequest(BaseModel):
    text: str
    topic:str

@app.post("/process/")
def process_endpoint(request: TextRequest):
    result = validate(request.text,request.topic)
    return {"processed_text": result}

@app.get("/")
def home():
    return {"message": "Welcome to the NLP API"}
