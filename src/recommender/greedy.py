import numpy as np

from src.recommender.base import BaseRecommender
from src.utils import cosine_similarity_matrix


class GreedyRecommender(BaseRecommender):
    def recommend(self, query_vec: np.ndarray, top_k: int = 10) -> np.ndarray:
        indices, embeddings = self._retrieval(query_vec)

        sim_query_to_item = cosine_similarity_matrix(query_vec, embeddings).squeeze()
        topk_indices = np.argsort(-sim_query_to_item)[:top_k]

        return indices[topk_indices]
