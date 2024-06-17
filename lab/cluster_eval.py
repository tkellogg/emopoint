import dataclasses
import pathlib

import numpy as np
import polars as pl
from sklearn import metrics, cluster
import streamlit as st

from emopoint import EmoModel
from train import Model

@dataclasses.dataclass
class ClusterRes:
    orig: float
    emopoint: float
    emoless: float


def measure(points: np.ndarray) -> float:
    mean = np.mean(points, axis=0)
    distances = np.linalg.norm(points - mean, axis=1)
    return np.mean(distances)


def eval_cluster(df: pl.DataFrame, emo_model: EmoModel) -> ClusterRes:
    embeddings = df["embedding"].to_numpy()
    emopoints = emo_model.emb_to_emo(embeddings)
    emoless = emo_model.remove_emo(embeddings)

    res = ClusterRes(
        orig=measure(embeddings),
        emopoint=measure(emopoints),
        emoless=measure(emoless),
    )

    return res


def main(model: Model):
    model_path = pathlib.Path(f"output/model-{model.label}.json")
    emo_model = EmoModel.from_json(model_path.read_text())

    expanded_df = pl.read_parquet(f"data/expanded_emotions-{model.label}.parquet")
    expanded_df = expanded_df.with_columns(pl.col("embedding").cast(pl.Array(pl.Float32, (model.num_dimensions,))))

    results = []
    for id in expanded_df.select(pl.col("id").unique())["id"]:
        results.append(
            eval_cluster(expanded_df.filter(pl.col("id") == pl.lit(id)), emo_model)
        )

    st.subheader("Original Embedding Space")
    st.line_chart(sorted([r.orig for r in results]))

    st.subheader("Emopoint Space")
    st.line_chart(sorted([r.emopoint for r in results]))

    st.subheader("Emotionless Embedding Space")
    st.line_chart(sorted([r.emoless for r in results]))