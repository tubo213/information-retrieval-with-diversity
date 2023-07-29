import faiss
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from const import MODEL_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.utils import save_pickle


def get_train_df():
    transaction_df = pl.read_csv(
        RAW_DATA_DIR / "transactions_train.csv", columns=["customer_id", "article_id", "price"]
    )
    negative_df = pl.read_csv(
        PROCESSED_DATA_DIR / "negative_sampling.csv",
    )

    labels = np.concatenate([np.ones(len(transaction_df)), np.zeros(len(negative_df))])
    train_df = pl.concat(
        [transaction_df.select(["customer_id", "article_id"]), negative_df]
    ).with_columns([pl.Series("label", labels)])

    return train_df


def label_encoding(train_df: pl.DataFrame, article_ids: np.ndarray, customer_ids: np.ndarray):
    article_id_to_idx = {article_id: idx for idx, article_id in enumerate(article_ids)}
    customer_id_to_idx = {customer_id: idx for idx, customer_id in enumerate(customer_ids)}
    train_df = train_df.with_columns(
        pl.col("customer_id").map_dict(customer_id_to_idx),
        pl.col("article_id").map_dict(article_id_to_idx),
    )
    encoded_article_ids = np.arange(len(article_id_to_idx))
    encoded_customer_ids = np.arange(len(customer_id_to_idx))

    return train_df, encoded_article_ids, encoded_customer_ids


class MFDataset(Dataset):
    def __init__(self, train_df: pl.DataFrame):
        self.article_ids = train_df.get_column("article_id").to_numpy()
        self.customer_ids = train_df.get_column("customer_id").to_numpy()
        self.labels = train_df.get_column("label").to_numpy()

    def __len__(self):
        return len(self.article_ids)

    def __getitem__(self, idx):
        customer_id = self.customer_ids[idx]
        article_id = self.article_ids[idx]
        label = self.labels[idx]

        return customer_id, article_id, label


class EmbeddingDataset(Dataset):
    def __init__(self, ids: np.ndarray):
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.ids[idx]


class MatrixFactorization(LightningModule):
    def __init__(self, num_customer: int, num_article: int, embedding_dim: int):
        super().__init__()
        self.customer_embedding = nn.Embedding(num_customer, embedding_dim)
        self.article_embedding = nn.Embedding(num_article, embedding_dim)

    def forward(self, customer_id: torch.Tensor, article_id: torch.Tensor):
        customer_embedding = self.customer_embedding(customer_id)
        article_embedding = self.article_embedding(article_id)
        sim = torch.sum(customer_embedding * article_embedding, dim=1)

        return sim

    def training_step(self, batch, batch_idx):
        customer_id, article_id, label = batch
        sim = self(customer_id, article_id)
        loss = nn.BCEWithLogitsLoss()(sim, label)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)

        return loss

    def embedding_customer(self, customer_id: torch.Tensor):
        return self.customer_embedding(customer_id)

    def embedding_article(self, article_id: torch.Tensor):
        return self.article_embedding(article_id)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


@torch.no_grad()
def get_article_embedding(model: MatrixFactorization, article_ids: np.ndarray):
    model = model.eval()
    article_ds = EmbeddingDataset(article_ids)
    article_dl = DataLoader(
        article_ds, batch_size=1024 * 15, shuffle=False, num_workers=24, pin_memory=True
    )

    article_embedding = []
    for article_id in tqdm(article_dl):
        article_embedding.append(model.embedding_article(article_id))

    return torch.cat(article_embedding, dim=0).cpu().numpy()


@torch.no_grad()
def get_user_embedding(model: MatrixFactorization, customer_ids: np.ndarray):
    model = model.eval()
    customer_ds = EmbeddingDataset(customer_ids)
    customer_dl = DataLoader(
        customer_ds, batch_size=1024 * 15, shuffle=False, num_workers=24, pin_memory=True
    )

    customer_embedding = []
    for customer_id in tqdm(customer_dl):
        customer_embedding.append(model.embedding_customer(customer_id))

    return torch.cat(customer_embedding, dim=0).cpu().numpy()


def main():
    output_dir = MODEL_DATA_DIR / "matrixfactorization"
    output_dir.mkdir(exist_ok=True, parents=True)
    train_df = get_train_df()
    article_ids = np.load(PROCESSED_DATA_DIR / "article_ids.npy", allow_pickle=True)
    customer_ids = np.load(PROCESSED_DATA_DIR / "customer_ids.npy", allow_pickle=True)

    train_df, encoded_article_ids, encoded_customer_ids = label_encoding(
        train_df, article_ids, customer_ids
    )

    batch_size = 1024 * 20
    embedding_dim = 128
    epochs = 10
    num_workers = 24

    mf_dataset = MFDataset(train_df)
    mf_dataloader = DataLoader(
        mf_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    model = MatrixFactorization(len(customer_ids), len(article_ids), embedding_dim)
    trainer = Trainer(accelerator="gpu", max_epochs=epochs, output_dir=output_dir)
    trainer.fit(model, mf_dataloader)

    article_embedding = get_article_embedding(model, encoded_article_ids)
    customer_embedding = get_user_embedding(model, encoded_customer_ids)

    # construct faiss index
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(embedding_dim))
    index.add_with_ids(article_embedding, article_ids)
    faiss.write_index(index, str(output_dir / "faiss.index"))

    # save user embedding
    customer2vec = dict(zip(customer_ids, customer_embedding))
    save_pickle(customer2vec, output_dir / "customer2vec.pkl")


if __name__ == "__main__":
    main()
