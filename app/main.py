from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from app.embedding import get_embedding, initialize_model
from sentence_transformers import SentenceTransformer
import os

app = FastAPI(title="Embedding API")

# Initialize model
model: Optional[SentenceTransformer] = None

def get_model():
    global model
    if model is None:
        try:
            model = initialize_model()
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load model: {str(e)}"
            )
    return model

class TextInput(BaseModel):
    text: str

class EmbeddingResponse(BaseModel):
    embedding: List[float]

@app.post("/embed", response_model=EmbeddingResponse)
async def create_embedding(input_data: TextInput):
    try:
        current_model = get_model()
        embedding = get_embedding(input_data.text, model=current_model)
        return EmbeddingResponse(embedding=embedding)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    try:
        get_model()
        return {"status": "healthy", "model_loaded": True}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e), "model_loaded": False}