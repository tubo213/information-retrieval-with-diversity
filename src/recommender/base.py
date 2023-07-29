from abc import abstractmethod

import numpy as np
import streamlit as st


class BaseRecommender:
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
