from typing import Set

import numpy as np
import polars as pl

from const import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.utils import save_json


def negative_sampling(
    transaction_df: pl.DataFrame,
    article_ids: np.ndarray,
    customer_ids: np.ndarray,
    num_sample_per_customer: int = 100,
) -> pl.DataFrame:
    """Generate negative sampling pairs for recommendation system.

    Args:
        transaction_df (pl.DataFrame): DataFrame containing transaction data.
        article_ids (np.ndarray): Array of article IDs.
        customer_ids (np.ndarray): Array of customer IDs.
        num_sample_per_customer (int, optional): Number of negative samples per customer. Defaults to 100.

    Returns:
        pl.DataFrame: DataFrame containing negative sampling pairs.
    """

    positive_pairs: Set[tuple] = set(
        zip(
            transaction_df.get_column("customer_id").to_numpy(),
            transaction_df.get_column("article_id").to_numpy(),
        )
    )
    random_article = np.random.choice(
        article_ids, size=(len(customer_ids), num_sample_per_customer)
    )

    negative_pairs: Set[tuple] = set(
        [
            (customer_id, article_id)
            for customer_id, article_ids in zip(customer_ids, random_article)
            for article_id in article_ids
        ]
    )
    negative_pairs = negative_pairs - positive_pairs
    negative_df = pl.DataFrame(
        {
            "customer_id": [x[0] for x in negative_pairs],
            "article_id": [x[1] for x in negative_pairs],
        }
    )

    return negative_df


def main():
    transaction_df = pl.read_csv(RAW_DATA_DIR / "transactions_train.csv")
    article_df = pl.read_csv(RAW_DATA_DIR / "articles.csv")

    # Customer IDs
    customer_ids = transaction_df["customer_id"].unique().sort().to_numpy()
    # Article IDs
    article_ids = transaction_df["article_id"].unique().sort().to_numpy()
    # Article ID to name
    article_id2name = dict(article_df[["article_id", "prod_name"]].to_numpy())
    # Negative sampling
    negative_df = negative_sampling(transaction_df, article_ids, customer_ids)

    print("Negative sampling rate: ", len(negative_df) / len(transaction_df))

    # save
    np.save(PROCESSED_DATA_DIR / "customer_ids.npy", customer_ids)
    np.save(PROCESSED_DATA_DIR / "article_ids.npy", article_ids)
    save_json(article_id2name, PROCESSED_DATA_DIR / "article_id2name.json")
    negative_df.write_csv(PROCESSED_DATA_DIR / "negative.csv")


if __name__ == "__main__":
    main()
