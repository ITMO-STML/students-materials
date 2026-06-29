import numpy as np

class MinHashGenerator:
    def __init__(self, num_hashes: int = 100, seed: int = 42):
        self.num_hashes = num_hashes
        self.seed = seed
        self.hash_functions = self._generate_hash_functions()
    
    def _generate_hash_functions(self):
        """Generate random hash functions: a*x + b mod prime"""
        np.random.seed(self.seed)
        # Use a large prime number
        prime = (1 << 32) - 5  # A large prime near 2^32
        
        a_values = np.random.randint(1, prime, self.num_hashes)
        b_values = np.random.randint(0, prime, self.num_hashes)
        
        return list(zip(a_values, b_values))
    
    def _hash_function(self, x: int, a: int, b: int, prime: int) -> int:
        """Apply hash function: (a*x + b) % prime"""
        return (a * x + b) % prime
    
    def compute_signature(self, shingles: set[int]) -> list[int]:
        """Compute MinHash signature for a set of shingles"""
        prime = (1 << 32) - 5
        signature = [float('inf')] * self.num_hashes
        
        for shingle in shingles:
            for i, (a, b) in enumerate(self.hash_functions):
                hash_val = self._hash_function(shingle, a, b, prime)
                if hash_val < signature[i]:
                    signature[i] = hash_val
        
        return signature
    
    def estimate_jaccard(self, sig1: list[int], sig2: list[int]) -> float:
        """Estimate Jaccard similarity from MinHash signatures"""
        if len(sig1) != len(sig2):
            raise ValueError("Signatures must have same length")
        
        matches = sum(1 for i in range(len(sig1)) if sig1[i] == sig2[i])
        return matches / len(sig1)