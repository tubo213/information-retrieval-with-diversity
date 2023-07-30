from pathlib import Path
from typing import Dict, List

import numpy as np
import polars as pl
import streamlit as st
from PIL import Image, ImageDraw
from pytorch_pfn_extras.config import Config

from const import CONF_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.recommender import BaseRecommender
from src.type import types
from src.utils import load_config, load_json


@st.cache_resource
def load_model(_config: Config) -> List[BaseRecommender]:
    recommenders = _config["/recommenders"]

    return recommenders


@st.cache_data
def load_metadata():
    article_df = pl.read_csv(RAW_DATA_DIR / "articles.csv")
    article_id2name = load_json(PROCESSED_DATA_DIR / "article_id2name.json")
    article_ids = np.load(PROCESSED_DATA_DIR / "article_ids.npy", allow_pickle=True)

    return article_df, article_id2name, article_ids


def init_article_id(article_id):
    st.session_state.article_id = article_id


def select_random_article(article_ids: np.ndarray) -> str:
    """Selects a random article ID from the given array.

    Args:
        article_ids (np.ndarray): An array of article IDs.

    Returns:
        str: The randomly selected article ID.

    Example:
        >>> article_ids = np.array([1, 2, 3, 4, 5])
        >>> select_random_article(article_ids)
        3
    """
    if st.button("Pick a random article"):
        st.session_state.article_id = np.random.choice(article_ids)
        article_id = st.session_state.article_id
        return article_id
    else:
        return st.session_state.article_id


class Renderer:
    def __init__(self, article_df: pl.DataFrame, article_id2name: Dict[str, str], img_dir: Path):
        self.article_df = article_df
        self.article_id2name = article_id2name
        self.img_dir = img_dir

    def display_img(self, article_id: int, width=None) -> None:
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
        if width is None:
            st.image(
                img,
                use_column_width=True,
                caption=name,
            )
        else:
            st.image(img, width=width, caption=name)

    def display_meta(self, article_id: int) -> None:
        display_cols = [
            "article_id",
            "prod_name",
            "product_type_name",
            "product_group_name",
            "graphical_appearance_name",
            "colour_group_name",
            "department_name",
            "index_name",
            "section_name",
            "garment_group_name",
            "detail_desc",
        ]
        article_meta = (
            self.article_df.select(display_cols)
            .filter(pl.col("article_id") == article_id)
            .to_pandas()
            .T
        )
        article_meta.columns = [""]
        st.table(article_meta)

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


def display_recommended_articles(
    query_vec: np.ndarray,
    recommender: BaseRecommender,
    renderer: Renderer,
    top_k: int = 10,
    max_display_per_col: int = 5,
):
    """Display recommended articles based on customer vector.

    Args:
        customer_vec (np.ndarray): The customer vector used for recommendation.
        recommender (BaseRecommender): The recommender object used for recommendation.
        renderer (Renderer): The renderer object used to display the articles.
        top_k (int, optional): The number of top recommended articles to display. Defaults to 10.
        max_display_per_col (int, optional): The maximum number of articles to display per column. Defaults to 5.

    Returns:
        None
    """
    # embeddingを入力として、おすすめの商品を取得
    article_ids = recommender.recommend(query_vec, top_k=top_k)

    # おすすめの商品を横並びに表示
    col = st.columns(max_display_per_col)
    for i, article_id in enumerate(article_ids):
        j = i % max_display_per_col
        with col[j]:
            renderer.display_img(article_id)


def main():
    config = load_config(CONF_DIR / "app.yaml", types=types)
    st.set_page_config(page_title="Recommender", page_icon="📦", layout="wide")
    with st.spinner("Loading model..."):
        recommenders = load_model(config)
        article_df, article_id2name, article_ids = load_metadata()
        renderer = Renderer(article_df, article_id2name, Path("data/raw/images"))

    st.title(f"H&M Recommender")
    st.markdown(f"Embeding model: {config['model_name']}")

    # Query Itemを初期化
    if "article_id" not in st.session_state:
        init_article_id(np.random.choice(article_ids))

    # ランダムにQueryを選択
    article_id = select_random_article(article_ids)

    # Query Itemを表示
    st.markdown("## Query Item")
    col1, col2 = st.columns([1, 3])
    with col1:
        renderer.display_img(article_id)
    with col2:
        renderer.display_meta(article_id)

    # # article idのembeddingを取得
    article_vec = recommenders[0].index.reconstruct(int(article_id)).reshape(1, -1)

    # おすすめの商品を表示
    st.markdown("## Recommended Items")
    for recommender in recommenders:
        st.markdown(f"### {recommender.__class__.__name__}")
        st.markdown("##### Parameters")
        recommender.set_params()
        display_recommended_articles(
            article_vec,
            recommender,
            renderer,
            top_k=config["top_k"],
            max_display_per_col=config["max_display_per_col"],
        )


if __name__ == "__main__":
    main()
