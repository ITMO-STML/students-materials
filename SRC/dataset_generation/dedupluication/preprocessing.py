import re
from dataclasses import dataclass
from typing import List, Set
import mmh3  # MurmurHash for hashing
import numpy as np

@dataclass
class Caption:
    id: int
    text: str
    shingles: Set[int]
    minhash_signature: List[int]

def preprocess_text(text: str) -> str:
    """Clean and normalize text"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text.strip()

def create_shingles(text: str, shingle_size: int = 3) -> Set[int]:
    """Convert text to hashed shingles (n-grams)"""
    words = text.split()
    shingles = set()
    
    for i in range(len(words) - shingle_size + 1):
        shingle = ' '.join(words[i:i + shingle_size])
        # Hash the shingle to a 32-bit integer
        shingle_hash = mmh3.hash(shingle)
        shingles.add(shingle_hash)
    
    return shingles