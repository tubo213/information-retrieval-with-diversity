import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import polars as pl
import streamlit as st
from PIL import Image, ImageDraw
from pytorch_pfn_extras.config import Config

from const import CONF_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.recommender import BaseRecommender
from src.type import types
from src.utils import load_config, load_json, load_pickle


@st.cache_resource
def load_model(_config: Config) -> Tuple[List[BaseRecommender], Dict[str, np.ndarray]]:
    recommenders = _config["/recommenders"]
    customer2vec = load_pickle(_config["/customer2vec_path"])

    return recommenders, customer2vec


@st.cache_data
def load_metadata():
    transaction_df = pl.scan_csv(RAW_DATA_DIR / "transactions_train.csv")
    article_id2name = load_json(PROCESSED_DATA_DIR / "article_id2name.json")
    customer_ids = np.load(PROCESSED_DATA_DIR / "customer_ids.npy", allow_pickle=True)

    return transaction_df, article_id2name, customer_ids


def init_customer_id(customer_id):
    st.session_state.customer_id = customer_id


def select_random_customer(customer_ids: np.ndarray) -> str:
    """Selects a random customer from a given array of customer IDs.

    Args:
        customer_ids (np.ndarray): An array of customer IDs.

    Returns:
        str: The randomly selected customer ID.
    """

    if st.button("Pick a random customer"):
        st.session_state.customer_id = np.random.choice(customer_ids)
        customer_id = st.session_state.customer_id
        return customer_id
    else:
        return st.session_state.customer_id


class Renderer:
    def __init__(self, article_id2name: Dict[str, str], img_dir: Path):
        self.article_id2name = article_id2name
        self.img_dir = img_dir

    def render(self, article_id: int) -> None:
        """Renders an article with the given article ID.

        Args:
            article_id (int): The ID of the article to render.

        Returns:
            None

        Raises:
            KeyError: If the article ID is not found in the article ID to name mapping.
        """

        name = self.article_id2name[str(article_id)]
        img = self._get_img(article_id)
        st.image(img, use_column_width=True, caption=name)

    def _get_img(self, article_id: int) -> Image:
        img_id = str(article_id).zfill(10)
        img_path = self.img_dir / img_id[:3] / f"{img_id}.jpg"
        if img_path.exists():
            img = Image.open(img_path)
        else:
            # 画像が存在しない場合は、No Image画像を表示
            img = Image.new("RGB", (64, 64), color=(255, 255, 255))
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), "No Image", fill=(0, 0, 0))

        return img


def display_rescent_articles(
    customer_id: str,
    transaction_df: pl.LazyFrame,
    renderer: Renderer,
    num_articles: int = 7,
):
    recent_articles = (
        transaction_df.filter(pl.col("customer_id") == customer_id)[:num_articles]
        .select("article_id")
        .collect()
        .to_numpy()
        .tolist()
    )
    col = st.columns(num_articles)
    for i, article_id in enumerate(recent_articles):
        with col[i]:
            renderer.render(article_id[0])


def display_recommended_items(
    customer_vec: np.ndarray,
    recommender: BaseRecommender,
    renderer: Renderer,
    top_k: int = 10,
    max_display_per_col: int = 5,
):
    """Display recommended items to the customer.

    Args:
        customer_vec (np.ndarray): The embedding of the customer.
        recommender (BaseRecommender): The recommender object used to generate recommendations.
        renderer (Renderer): The renderer object used to display the recommended items.
        top_k (int, optional): The number of recommended items to display. Defaults to 10.
        max_display_per_col (int, optional): The maximum number of items to display per column. Defaults to 5.

    Returns:
        None
    """

    # embeddingを入力として、おすすめの商品を取得
    article_ids = recommender.recommend(customer_vec, top_k=top_k)

    # おすすめの商品を横並びに表示
    col = st.columns(max_display_per_col)
    for i, article_id in enumerate(article_ids):
        j = i % max_display_per_col
        with col[j]:
            renderer.render(article_id)


def main():
    config = load_config(CONF_DIR / "app.yaml", types=types)
    st.set_page_config(page_title="Recommender", page_icon="📦", layout="wide")
    with st.spinner("Loading model..."):
        recommenders, customer2vec = load_model(config)
        transaction_df, article_id2name, customer_ids = load_metadata()
        renderer = Renderer(article_id2name, Path("data/raw/images"))

    st.title("H&M Recommender")

    # 顧客を初期化
    if "customer_id" not in st.session_state:
        init_customer_id(customer_ids[0])
    # ランダムに顧客を選択
    customer_id = select_random_customer(customer_ids)

    # 顧客の最近購入した商品を表示
    st.markdown("## Items recently purchased by the customer")
    display_rescent_articles(customer_id, transaction_df, renderer, num_articles=config["top_k"])

    # customer idからembeddingを取得
    customer_vec = customer2vec[customer_id][None, :]

    # おすすめの商品を表示
    st.markdown("## Recommended Items")
    for recommender in recommenders:
        st.markdown(f"### {recommender.__class__.__name__}")
        st.markdown("##### Parameters")
        recommender.set_params()
        display_recommended_items(
            customer_vec,
            recommender,
            renderer,
            top_k=config["top_k"],
            max_display_per_col=config["max_display_per_col"],
        )


if __name__ == "__main__":
    main()
