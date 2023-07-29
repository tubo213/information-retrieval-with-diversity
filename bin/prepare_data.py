import json
from pathlib import Path

import numpy as np
import polars as pl

from const import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.utils import save_json


def main():
    transaction_df = pl.read_csv(RAW_DATA_DIR / "transactions_train.csv")
    article_df = pl.read_csv(RAW_DATA_DIR / "articles.csv")

    # Customer IDs
    customer_ids = transaction_df["customer_id"].unique().sort().to_numpy()
    # Article IDs
    article_ids = transaction_df["article_id"].unique().sort().to_numpy()
    # Article ID to name
    article_id2name = dict(article_df[["article_id", "prod_name"]].to_numpy())

    # save
    np.save(PROCESSED_DATA_DIR / "customer_ids.npy", customer_ids)
    np.save(PROCESSED_DATA_DIR / "article_ids.npy", article_ids)
    save_json(article_id2name, PROCESSED_DATA_DIR / "article_id2name.json")


if __name__ == "__main__":
    main()
