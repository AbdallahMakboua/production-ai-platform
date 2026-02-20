from typing import List
import os
from dotenv import load_dotenv

load_dotenv()

class TextChunker:
    """Split text into overlapping chunks."""
    
    def __init__(self, chunk_size: int = None, overlap: int = None):
        self.chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE", 512))
        self.overlap = overlap or int(os.getenv("CHUNK_OVERLAP", 50))
        if self.overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
    
    def chunk(self, text: str) -> List[str]:
        """Split text into chunks with overlap.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []
        
        words = text.split()
        
        if len(words) <= self.chunk_size:
            return [text]
        
        chunks = []
        i = 0
        
        while i < len(words):
            end = min(i + self.chunk_size, len(words))
            chunk = " ".join(words[i:end])
            chunks.append(chunk)
            if end == len(words):
                break
            i = end - self.overlap
        
        return chunks