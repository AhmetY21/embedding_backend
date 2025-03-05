import os
from sentence_transformers import SentenceTransformer

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
MODEL_CACHE_DIR = os.getenv('MODEL_CACHE_DIR', '/app/model_cache')

def initialize_model():
    # Ensure cache directory exists
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    
    # Initialize model with cache
    return SentenceTransformer(MODEL_NAME, cache_folder=MODEL_CACHE_DIR)

def get_embedding(search_name, model=None):
    if model is None:
        model = initialize_model()
    return model.encode([search_name])[0].tolist()