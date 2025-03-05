from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from threading import Lock
import os
import logging

# --- Configuration ---
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CACHE_PATH = "/persistent_volume/models"  # Render persistent storage
API_KEY = os.getenv("API_KEY")  # Set in Render dashboard
MAX_RATE = "100/minute"  # Adjust based on your Render plan

# --- Initialization ---
app = FastAPI(title="Embedding API")
logger = logging.getLogger("uvicorn.error")
model_lock = Lock()
model = None

# --- Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- Model Handling ---
def initialize_model():
    try:
        logger.info("Initializing Sentence Transformer model...")
        return SentenceTransformer(
            MODEL_NAME,
            cache_folder=os.getenv("MODEL_CACHE", DEFAULT_CACHE_PATH)
        )
    except Exception as e:
        logger.error(f"Model initialization failed: {str(e)}")
        raise

def get_model():
    global model
    with model_lock:
        if model is None:
            model = initialize_model()
    return model

# --- Security Middleware ---
@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    if request.url.path in ["/health", "/metrics", "/docs", "/openapi.json"]:
        return await call_next(request)
        
    if request.headers.get("X-API-Key") != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    return await call_next(request)

# --- Data Models ---
class TextInput(BaseModel):
    text: str

class EmbeddingResponse(BaseModel):
    embedding: List[float]

# --- Endpoints ---
@app.post("/embed", response_model=EmbeddingResponse)
@limiter.limit(MAX_RATE)
async def create_embedding(
    request: Request,
    input_data: TextInput,
    current_model: SentenceTransformer = Depends(get_model)
):
    try:
        if not input_data.text.strip():
            raise HTTPException(status_code=400, detail="Empty input text")
            
        embedding = current_model.encode(
            input_data.text,
            convert_to_tensor=False,
            normalize_embeddings=True
        ).tolist()
        
        return EmbeddingResponse(embedding=embedding)
        
    except Exception as e:
        logger.error(f"Embedding error: {str(e)}")
        raise HTTPException(status_code=500, detail="Embedding generation failed")

@app.get("/health")
async def health_check():
    try:
        model_status = "loaded" if model else "uninitialized"
        return {
            "status": "healthy",
            "model": model_status,
            "model_dimensions": model.get_sentence_embedding_dimension() if model else 0,
            "system_load": os.getloadavg(),
            "rate_limit": MAX_RATE
        }
    except Exception as e:
        return {"status": "degraded", "error": str(e)}

# --- Optional Metrics Endpoint (Requires prometheus_fastapi_instrumentator) ---
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    Instrumentator().instrument(app).expose(app)
except ImportError:
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
