import faiss
import numpy as np

from src.recommender.base import BaseRecommender
from src.utils import cosine_similarity_matrix


class GreedyRecommender(BaseRecommender):
    def __init__(self, index_path: str, num_candidates: int = 100):
        """Greedy recommender

        Args:
            index_path (str): Path to the index file
            num_candidates (int, optional): Defaults to 100.
        """
        self.index = faiss.read_index(index_path)
        self.num_candidates = num_candidates

    def recommend(self, query_vec: np.ndarray, top_k: int = 10) -> np.ndarray:
        _, indices, embeddings = self.index.search_and_reconstruct(query_vec, self.num_candidates)
        indices = indices.squeeze()
        embeddings = embeddings.squeeze()

        sim_query_to_item = cosine_similarity_matrix(query_vec, embeddings).squeeze()
        topk_indices = np.argsort(-sim_query_to_item)[:top_k]

        return indices[topk_indices]
