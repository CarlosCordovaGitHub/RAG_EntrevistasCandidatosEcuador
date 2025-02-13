class Config:
    CHROMA_DB_PATH = "./chroma_db"
    MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    OLLAMA_MODEL = "llama3.2:latest"
    BATCH_SIZE = 100
    MAX_CHUNK_LENGTH = 512