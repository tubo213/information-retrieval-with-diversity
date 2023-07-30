import faiss
import numpy as np
import polars as pl
from gensim.models import word2vec

from const import CONF_DIR, MODEL_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.utils import load_config, save_pickle


def main():
    config = load_config(CONF_DIR / "item2vec.yaml")
    # set output dir
    output_dir = MODEL_DATA_DIR / "item2vec"
    output_dir.mkdir(parents=True, exist_ok=True)

    # load data
    transactions_df = pl.read_csv(RAW_DATA_DIR / "transactions_train.csv")
    article_ids = np.load(PROCESSED_DATA_DIR / "article_ids.npy")

    # purchase history of each customer
    item_sentences = transactions_df.groupby("customer_id").agg(pl.col("article_id"))

    # train
    model = word2vec.Word2Vec(
        item_sentences.get_column("article_id").to_list(), **config["model_params"]
    )
    ## save model
    model.save(str(output_dir / "word2vec.model"))

    # construct faiss index
    emb_dim = model.wv.vector_size
    article_vecs = {article_id: model.wv[article_id] for article_id in article_ids}
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(emb_dim))
    vecs = np.array(list(article_vecs.values()))
    index.add_with_ids(vecs, article_ids)
    faiss.write_index(index, str(output_dir / "faiss.index"))

    # save user embedding
    customer2vec = {}
    for customer_id, sentence in item_sentences.iter_rows():
        customer_vec = np.mean(
            [model.wv[article_id] for article_id in sentence], axis=0
        )  # mean pooling
        customer2vec[customer_id] = customer_vec
    save_pickle(customer2vec, output_dir / "customer2vec.pkl")


if __name__ == "__main__":
    main()
