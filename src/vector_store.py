import faiss
import numpy as np


class VectorStore:
    """
    Wraps a FAISS index together with the chunk metadata (text + timestamp)
    so that when FAISS gives us back a match, we know what text/timestamp
    it corresponds to (FAISS itself only knows about numbers, not our data).
    """

    def __init__(self, embedding_dim: int = 384):
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.chunks = []  # parallel list: chunks[i] corresponds to vector i in the index

    def add(self, embeddings: np.ndarray, chunks: list[dict]):
        """
        Adds embeddings + their corresponding chunk metadata to the store.
        """
        # FAISS requires float32 specifically
        embeddings = np.asarray(embeddings, dtype="float32")
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> list[dict]:
        """
        Finds the top_k most similar chunks to the query embedding.
        Returns a list of chunk dicts, each with an added "score" key.
        """
        query_embedding = np.asarray([query_embedding], dtype="float32")
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue  # FAISS pads with -1 if fewer than top_k results exist
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(score)
            results.append(chunk)

        return results

    def __len__(self):
        return len(self.chunks)


def retrieve_relevant_chunks(store: VectorStore, query: str, top_k: int = 4) -> list[dict]:
    """
    Given a user's question, finds the top_k most relevant transcript
    chunks from the vector store.

    This is what "the retriever" means in RAG: turning a question into
    an embedding, then finding the stored chunks whose embeddings are
    closest to it in meaning.
    """
    from src.embeddings import embed_query  # local import avoids circular import issues

    query_embedding = embed_query(query)
    return store.search(query_embedding, top_k=top_k)