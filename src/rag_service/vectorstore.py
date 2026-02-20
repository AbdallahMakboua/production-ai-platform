import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict
import os
import json
from dotenv import load_dotenv

load_dotenv()


class FAISSVectorStore:
    """FAISS-backed vector store for semantic search."""
    
    def __init__(self, model_name: str = None, vector_dim: int = None, model=None):
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.vector_dim = vector_dim or int(os.getenv("VECTOR_DIM", 384))
        self.dimension = self.vector_dim
        
        # Load embedding model
        self.model = model or SentenceTransformer(self.model_name)
        
        # Create FAISS index (L2 = Euclidean distance)
        self.index = faiss.IndexFlatL2(self.vector_dim)
        
        # Metadata store: id -> text mapping
        self.metadata = {}
        self.id_list = []  # Track order of IDs
    
    def add_texts(self, texts: List[str], ids: List[str]) -> None:
        """Add texts to vector store.
        
        Args:
            texts: List of text strings
            ids: List of document IDs
        """
        if not texts or not ids:
            return
        
        # Embed texts
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        # Normalize embeddings for L2 distance
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings.astype(np.float32))
        
        # Store metadata
        for text_id, text in zip(ids, texts):
            self.metadata[text_id] = text
            self.id_list.append(text_id)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar texts.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of results with id, text, and score
        """
        if self.index.ntotal == 0:
            return []
        
        # Embed query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search index
        distances, indices = self.index.search(
            query_embedding.astype(np.float32), 
            min(top_k, self.index.ntotal)
        )
        
        # Build results (IMPORTANT: convert distance to similarity score)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:  # Invalid index
                continue
            
            # L2 distance to cosine similarity conversion
            l2_distance = float(distances[0][i])
            # For normalized vectors: cosine_similarity = 1 - (l2_distance / 2)
            similarity_score = max(0, 1 - (l2_distance / 2))
            
            doc_id = self.id_list[int(idx)]
            
            results.append({
                "id": doc_id,
                "text": self.metadata[doc_id],
                "score": similarity_score,
                "rank": len(results) + 1
            })
        
        return results
    
    def persist(self, path: str) -> None:
        """Save index and metadata to disk.
        
        Args:
            path: Directory path to save to
        """
        os.makedirs(path, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        
        # Save metadata
        metadata_file = os.path.join(path, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump({
                "metadata": self.metadata,
                "id_list": self.id_list,
                "model_name": self.model_name
            }, f)
    
    def load(self, path: str) -> None:
        """Load index and metadata from disk.
        
        Args:
            path: Directory path to load from
        """
        # Load FAISS index
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))
        
        # Load metadata
        metadata_file = os.path.join(path, "metadata.json")
        with open(metadata_file, 'r') as f:
            data = json.load(f)
            self.metadata = data["metadata"]
            self.id_list = data["id_list"]
            self.model_name = data["model_name"]