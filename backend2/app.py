from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
import os
import shutil
from services import maine


app = FastAPI()

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="rag_docs")

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")


class QueryRequest(BaseModel):
    query: str

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        file_location = f"temp/{file.filename}"
        os.makedirs("temp", exist_ok=True)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/")
def query_rag(request: QueryRequest):
    # query_embedding = embed_model.encode(request.query)
    # results = collection.query(embedding=query_embedding.tolist(), n_results=3)
    
    # # Extract top document
    # top_doc = results["documents"][0] if results["documents"] else ""
    
    # # Chat completion
    # response = chat_model.predict(f"Context: {top_doc}\nUser: {request.query}\nAI:")
    print(request.query)
    ret , gen ,out = maine(request.query)
    
    return {"retrival evaluation": ret,"generation evaluation":gen,"final output" :out }

@app.get("/")
def home():
    return {"message": "Welcome to the NLP API"}

