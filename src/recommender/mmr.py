import numpy as np
import streamlit as st

from src.recommender.base import BaseRecommender
from src.utils import cosine_similarity_matrix


class MMRRecommender(BaseRecommender):
    def __init__(self, index_path: str, lam: float = 0.5, num_candidates: int = 100):
        super().__init__(index_path, num_candidates)
        self.lam = lam

    def recommend(self, query_vec: np.ndarray, top_k: int = 10) -> np.ndarray:
        indices, embeddings = self._retrieval(query_vec)

        # MMR
        num_items = embeddings.shape[0]
        unselected = list(range(num_items))
        sim_query_to_item = cosine_similarity_matrix(
            query_vec, embeddings
        ).squeeze()  # クエリとアイテムの類似度
        item_sim_mat = cosine_similarity_matrix(embeddings, embeddings)  # アイテムの類似度行列

        # 1st iteration
        select_idx = np.argmax(sim_query_to_item)
        selected = [select_idx]
        unselected.remove(select_idx)

        # other iterations
        for _ in range(top_k - 1):
            sim_to_query = sim_query_to_item[unselected]  # (num_unselected,)
            sim_between_items = np.max(
                item_sim_mat[unselected][:, selected], axis=1
            )  # (num_unselected,)
            mmr = self.lam * sim_to_query - (1 - self.lam) * sim_between_items
            mmr_index = unselected[np.argmax(mmr)]

            selected.append(mmr_index)
            unselected.remove(mmr_index)

        return indices[selected]

    def set_params(self):
        text = """$\lambda$ (control trade-off between relevance and diversity, larger $\lambda$ means more relevance and less diversity)"""
        self.lam = st.slider(text, 0.0, 1.0, 0.3, 0.05)
