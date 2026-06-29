import mmh3

class LSHIndex:
    def __init__(self, signature_length: int, bands: int, rows_per_band: int):
        self.bands = bands
        self.rows_per_band = rows_per_band
        self.tables = [dict() for _ in range(bands)]
        
        if bands * rows_per_band != signature_length:
            raise ValueError("bands * rows_per_band must equal signature length")
    
    def add_signature(self, caption_id: int, signature: list[int]):
        """Add a signature to the LSH index"""
        for band in range(self.bands):
            # Extract the band segment
            start = band * self.rows_per_band
            end = start + self.rows_per_band
            band_signature = tuple(signature[start:end])
            
            # Create bucket key by hashing the band signature
            bucket_key = mmh3.hash64(str(band_signature))[0]
            
            if bucket_key not in self.tables[band]:
                self.tables[band][bucket_key] = []
            
            self.tables[band][bucket_key].append(caption_id)
    
    def get_candidate_pairs(self) -> set[tuple]:
        """Find all candidate duplicate pairs"""
        candidate_pairs = set()
        
        for band_table in self.tables:
            for bucket in band_table.values():
                if len(bucket) > 1:
                    # Add all pairs from this bucket
                    for i in range(len(bucket)):
                        for j in range(i + 1, len(bucket)):
                            candidate_pairs.add((min(bucket[i], bucket[j]), 
                                               max(bucket[i], bucket[j])))
        
        return candidate_pairs
