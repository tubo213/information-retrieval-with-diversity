from abc import abstractmethod

import faiss
import numpy as np
import streamlit as st


class BaseRecommender:
    def __init__(self, index_path: str, num_candidates: int = 100):
        """Base recommender

        Args:
            index_path (str): Path to the index file
            num_candidates (int, optional): Defaults to 100.
        """
        self.index = faiss.read_index(index_path)
        self.num_candidates = num_candidates

    @abstractmethod
    def recommend(self, customer_vec: np.ndarray, top_k: int = 10) -> np.ndarray:
        """Recommends top-k items to a customer based on their vector representation.

        Args:
            customer_vec (np.ndarray): The vector representation of the customer.
            top_k (int, optional): The number of items to recommend. Defaults to 10.

        Returns:
            np.ndarray: An array of recommended items.

        Raises:
            NotImplementedError: This method is not implemented yet.
        """

        raise NotImplementedError

    def set_params(self) -> None:
        """Sets the parameters for the function.

        This function does not take any parameters.

        Returns:
            None: This function does not return anything.
        """

        st.write("No parameters to set.")

    def _retrieval(self, query_vec: np.ndarray) -> np.ndarray:
        _, indices, embeddings = self.index.search_and_reconstruct(query_vec, self.num_candidates)
        indices = indices.squeeze()
        embeddings = embeddings.squeeze()

        return indices, embeddings
