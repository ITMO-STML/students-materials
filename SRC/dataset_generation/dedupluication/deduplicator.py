from minhash import MinHashGenerator
from lsh import LSHIndex
from preprocessing import Caption, create_shingles, preprocess_text
import argparse
import json
from tqdm import tqdm

class MinHashDeduplicator:
    def __init__(self, num_hashes: int = 100, bands: int = 20, shingle_size: int = 3, similarity_threshold: float = 0.8) -> None:
        self.num_hashes = num_hashes
        self.bands = bands
        self.rows_per_band = num_hashes // bands
        self.shingle_size = shingle_size
        self.similarity_threshold = similarity_threshold

        self.minhash_gen = MinHashGenerator(num_hashes)
        self.lsh_index = LSHIndex(num_hashes, bands, self.rows_per_band)
        self.captions = {}

    def add_caption(self, caption_id: int, text: str) -> Caption:
        """Add a caption to the deduplication system"""
        processed_text = preprocess_text(text)
        shingles = create_shingles(processed_text, self.shingle_size)
        signature = self.minhash_gen.compute_signature(shingles)

        caption = Caption(caption_id, text, shingles, signature)
        self.captions[caption_id] = caption
        self.lsh_index.add_signature(caption_id, signature)

        return caption

    def find_duplicates(self) -> list[tuple]:
        """Find all duplicate caption pairs"""
        # Step 1: Get candidate pairs using LSH
        candidate_pairs = self.lsh_index.get_candidate_pairs()
        print(f"Found {len(candidate_pairs)} candidate pairs")

        # Step 2: Verify candidates using exact Jaccard similarity
        duplicate_pairs = []

        for id1, id2 in candidate_pairs:
            caption1 = self.captions[id1]
            caption2 = self.captions[id2]

            # Calculate exact Jaccard similarity
            intersection = len(caption1.shingles & caption2.shingles)
            union = len(caption1.shingles | caption2.shingles)

            if union > 0:  # Avoid division by zero
                jaccard_similarity = intersection / union

                if jaccard_similarity >= self.similarity_threshold:
                    duplicate_pairs.append((id1, id2, jaccard_similarity))

        return duplicate_pairs

    def get_duplicate_clusters(self) -> list[set[int]]:
        """Group duplicates into connected clusters"""
        duplicate_pairs = self.find_duplicates()

        # Use union-find to create clusters
        parent = {}

        def find(x):
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            parent[find(x)] = find(y)
        
        # Union all duplicate pairs
        for id1, id2, _ in duplicate_pairs:
            union(id1, id2)
        
        # Group by root parent
        clusters = {}
        for caption_id in self.captions:
            root = find(caption_id)
            if root not in clusters:
                clusters[root] = set()
            clusters[root].add(caption_id)
        
        # Return only clusters with duplicates (size > 1)
        return [cluster for cluster in clusters.values() if len(cluster) > 1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True, help="Path to JSON with descriptions")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save filtered JSON")
    return parser.parse_args()

def gen_clusters(data: list[dict[str, str]], name: str | list[str], ids_to_rm: set[int] = set()) -> set[int]:
    dedup = MinHashDeduplicator(
        num_hashes=100,
        bands=20,
        shingle_size=3,
        similarity_threshold=0.1
    )
    dedup = MinHashDeduplicator(
        num_hashes=100,
        bands=20,
        shingle_size=3,
        similarity_threshold=0.1
    )
    # img disc dedup
    for caption_id, row in enumerate(tqdm(data, leave=False, desc="Dedup captioning")):
        if isinstance(name, list):
            desc = ""
            for n in name:
                desc = desc + row[n] + " "
        else:
            desc = row[name]
        desc = desc.strip()
        dedup.add_caption(caption_id,  desc)
    
    clusters = dedup.get_duplicate_clusters()

    for cluster in tqdm(clusters, leave=False, desc="ids storing"):
        ids_to_rm.update(sorted(list(cluster))[1:])
    return ids_to_rm
    
def main():
    args = parse_args()
    with open(args.json_path, encoding="utf-8") as file:
        data: list = json.load(file)
    
    ids_to_rm = gen_clusters(data, "img_desc")
    ids_to_rm = gen_clusters(data, "vid_desc", ids_to_rm)
    ids_to_rm = gen_clusters(data, ["img_desc", "vid_desc"], ids_to_rm)

    ids_to_rm = list(sorted(ids_to_rm, key=lambda x: -x))
    for idx in ids_to_rm:
        data.pop(idx)

    print(f"{len(ids_to_rm)} samples would be removed")
    with open(args.save_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

if __name__ == "__main__":
    main()