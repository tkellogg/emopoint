import dataclasses
import pathlib

import numpy as np
import polars as pl
from sklearn import metrics, cluster
import streamlit as st

from emopoint import EmoModel
from train import Model
from models import MODELS

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


def measure_model(model: Model) -> pl.DataFrame:
    model_path = pathlib.Path(f"output/model-{model.label}.json")
    emo_model = EmoModel.from_json(model_path.read_text())

    expanded_df = pl.read_parquet(f"data/expanded_emotions-{model.label}.parquet")
    expanded_df = expanded_df.with_columns(pl.col("embedding").cast(pl.Array(pl.Float32, (model.num_dimensions,))))

    results = []
    for id in expanded_df.select(pl.col("id").unique())["id"]:
        results.append(
            eval_cluster(expanded_df.filter(pl.col("id") == pl.lit(id)), emo_model)
        )

    return pl.DataFrame({
        "original": sorted([r.orig for r in results]),
        "emopoint": sorted([r.emopoint for r in results]),
        "emotionless": sorted([r.emoless for r in results]),
    })

def main(model: Model):
    res_df = measure_model(model)
    st.markdown("""
    ## Distance From Avg
    For each original text, we modified the emotion while keeping the core meaning as close as possible.
    I used an LLM to create altered texts. For each original text, I calculated the average distance from the 
    centroid.
                
    * `original` (red) — Bigger means the model captures more emotional information
    * `emotionless` (pale blue) — The error. How much non-emotion the was added (when instructed to only alter the emotion)
    * `emopoint` (dark blue) — Same as `original`, but converted to 3D `emopoint` space. The difference from `original` is
                explained by whatever happens in PCA.
    """)
    st.line_chart(res_df)

    st.subheader("Comparison of original between models")
    all_models_df = pl.DataFrame({
        m.label: measure_model(m)["original"]
        for m in MODELS
    })
    st.line_chart(all_models_df)

    st.subheader("Sans Emotion (amount of non-emotion picked up)")
    all_models_df = pl.DataFrame({
        m.label: measure_model(m)["emotionless"]
        for m in MODELS
    })
    st.line_chart(all_models_df)