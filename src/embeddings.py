from sentence_transformers import SentenceTransformer
import numpy as np

# Loaded once and reused - loading the model from disk/downloading it
# is slow, so we don't want to do it on every single function call.
_model = None


def get_embedding_model() -> SentenceTransformer:
    """
    Returns a cached SentenceTransformer model instance.
    Downloads the model from Hugging Face on first use (one-time,
    then it's cached locally on disk for future runs).
    """
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed_chunks(chunks: list[dict]) -> np.ndarray:
    """
    Generates embeddings for a list of chunk dicts (each with a "text" key).
    Returns a NumPy array of shape (num_chunks, embedding_dim).
    """
    model = get_embedding_model()
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return embeddings


def embed_query(query: str) -> np.ndarray:
    """
    Generates an embedding for a single user question, using the
    same model (critical - query and chunks must use the same
    embedding space to be comparable).
    """
    model = get_embedding_model()
    return model.encode([query], normalize_embeddings=True)[0]