import dataclasses
import functools
import os
import pathlib

import dotenv
import matplotlib.pyplot as plt
import numpy as np
import openai
from plotly import express as px, graph_objects as go
import polars as pl
from sklearn import metrics, linear_model
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import streamlit as st
import tiktoken as tk

from emopoint import EMOTIONS, EKMAN_MAP, DimLabel, EmoModel


dotenv.load_dotenv()


def progress_bar(max, start=None):
    if start is None:
        start = 0
    else:
        tmp = max
        max = start
        start = tmp

    bar = st.progress(0)
    for i in range(start, max):
        try:
            yield i
        except KeyboardInterrupt:
            return
        bar.progress(i / max)


@dataclasses.dataclass
class Model:
    id: str
    label: str
    opts: dict = dataclasses.field(default_factory=dict)
    max_tokens: int = 8191

    @property
    def path(self) -> pathlib.Path:
        return pathlib.Path(f"data/models/{self.label}.parquet")

    @property
    def num_dimensions(self) -> int:
        return self.opts.get("dimensions", 1536)

    def generate_df(self, raw_df: pl.DataFrame) -> pl.DataFrame:
        """
        Generate embeddings for all texts for this model.
        """
        enc = tk.encoding_for_model(self.id)
        oai = openai.OpenAI(api_key=os.environ["OPENAI_KEY"])
        
        def embed(texts: pl.Series) -> pl.Series:
            input_buf = []
            output_buf = []
            num_toks = 0
            for text in texts:
                toks = len(enc.encode_batch(text))
                if num_toks + toks > self.max_tokens:
                    print("Flushing", len(input_buf), "texts")
                    resp = oai.embeddings.create(input=input_buf, model=self.id, **self.opts)
                    output_buf.extend([d.embedding for d in resp.data])
                    num_toks = 0
                    input_buf.clear()
                num_toks += toks
                input_buf.append(text)

            if len(input_buf) > 0:
                print("Flushing", len(input_buf), "texts")
                resp = oai.embeddings.create(input=input_buf, model=self.id, **self.opts)
                output_buf.extend([d.embedding for d in resp.data])
            
            return pl.Series(output_buf)

        with st.spinner(text="Generating embeddings..."):
            dataset_df = raw_df.with_columns(
                pl.col("text").map_batches(embed, return_dtype=pl.List(pl.Float64)).alias("embedding")
            )
            self.path.parent.mkdir(parents=True, exist_ok=True)
            dataset_df.write_parquet(self.path, compression="snappy")
        return dataset_df



@st.cache_data
def load_embeddings(path: pathlib.Path) -> pl.DataFrame:
    return pl.read_parquet(path)

def map_to_ekman(df: pl.DataFrame) -> pl.DataFrame:
    """
    Map the embeddings to the EKMAN space.
    """
    return (
        df
        .select(
             pl.col("id"),
             pl.col("text"),
             pl.col("embedding"),
             *[
                 functools.reduce(
                     lambda acc, emotion: acc | pl.col(emotion),
                     emo_list,
                     pl.lit(False),
                 ).alias(ekman)
                 for ekman, emo_list in EKMAN_MAP.items()
             ]
         )
    )

@dataclasses.dataclass
class Dimension:
    label: str
    labels: tuple[str, str]
    pca: PCA
    reg: linear_model.LogisticRegression
    metrics: dict[str, float]

    @property
    def ordered_labels(self) -> tuple[str, str]:
        if self.metrics["positive"] >= self.metrics["negative"]:
            return self.labels
        else:
            return self.labels[::-1]

def main(model: Model):

    raw_df = pl.concat([
        pl.read_csv("data/goemotions_1.csv"),
        pl.read_csv("data/goemotions_2.csv"),
        pl.read_csv("data/goemotions_3.csv"),
    ])

    def empty(dataset_df: pl.DataFrame) -> pl.DataFrame:
        """
        Get the rows that contain "zero" emotion. We'll use this as the
        lower end of the spectrum to create variation.
        """
        return dataset_df.filter(functools.reduce(
            lambda acc, emotion: acc & (pl.col(emotion) == pl.lit(0)),
            [e for e in EMOTIONS if e != "neutral"],
            pl.col("neutral") == pl.lit(1),
        ))

    def slice(emotion: str, dataset_df: pl.DataFrame=raw_df) -> pl.DataFrame:
        return dataset_df.filter(pl.col(emotion) > 0)

    def concat(a: pl.DataFrame, b: pl.DataFrame, num_dims: int) -> np.ndarray:
        return pl.concat([a, b])["embedding"].cast(pl.Array(pl.Float32, (num_dims,))).to_numpy()


    def train_dimension(emotions: tuple[str, str], positive_df: pl.DataFrame, negative_df: pl.DataFrame, num_dims, num_components=1, plot=True) -> Dimension:
        """
        Calculate the dimension of a given emotion.
        """
        emotion = "_".join(emotions)

        if positive_df.shape[0] > negative_df.shape[0]:
            positive_df = positive_df.sample(negative_df.shape[0])
        elif positive_df.shape[0] < negative_df.shape[0]:
            negative_df = negative_df.sample(positive_df.shape[0])

        negative_df = negative_df.sample(positive_df.shape[0])
        positive_train_df, positive_test_df, negative_train_df, negative_test_df = train_test_split(positive_df, negative_df, train_size=0.8)
        
        train_arr = concat(positive_train_df, negative_train_df, num_dims)

        pca = PCA(n_components=num_components)
        pca.fit(train_arr)

        # signal_test_arr = pca.transform(signal_test_df["embedding"].cast(pl.Array(pl.Float32, (num_dims,))).to_numpy())
        # neutral_test_arr = pca.transform(negative_test_df["embedding"].cast(pl.Array(pl.Float32, (num_dims,))).to_numpy())
        positive_test_arr = (pca.components_[0].reshape(1, -1) @ positive_test_df["embedding"].cast(pl.Array(pl.Float32, (num_dims,))).to_numpy().T).T
        negative_test_arr = (pca.components_[0].reshape(1, -1) @ negative_test_df["embedding"].cast(pl.Array(pl.Float32, (num_dims,))).to_numpy().T).T

        if plot:
            fig, ax = plt.subplots()
            ax.hist(positive_test_arr, bins=30, alpha=0.5, label=emotions[0], edgecolor='black')
            ax.hist(negative_test_arr, bins=30, alpha=0.5, label=emotions[1], edgecolor='black')

            # Adding labels, title, and legend
            ax.set_title(f"Distribution along {emotion}")
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.legend()

        y_true = np.concatenate([np.ones(len(positive_test_arr)), np.zeros(len(negative_test_arr))])

        reg = linear_model.LogisticRegression()
        reg.fit(np.concatenate([positive_test_arr, negative_test_arr]), y_true)

        if plot:
            ax.axvline(reg.classes_[0], color='r', linestyle='--')
            # Display the histogram in Streamlit
            st.pyplot(fig)


        # signal_result_arr = np.array([x < 0 for x in signal_result_arr])
        # neutral_result_arr = np.array([x >= 0 for x in neutral_result_arr])
        positive_result_arr = reg.predict(positive_test_arr)
        negative_result_arr = reg.predict(negative_test_arr)
        
        y_pred = np.concatenate([positive_result_arr, negative_result_arr])

        # result_df = (
        #     pl.DataFrame({
        #         "true": y_true,
        #         "pred": y_pred,
        #         "id": np.concatenate([signal_test_df["id"], neutral_test_df["id"]]),
        #     })
        #     .filter(pl.col("pred") != pl.col("true"))
        #     .join(raw_df, on="id", how="inner")
        # )
        # st.table(pl.DataFrame({
        #     "count": pl.Series([result_df.filter(pl.col(emo) == 1).count().rows()[0][0] for emo in EMOTIONS]),
        #     "emotion": pl.Series(EMOTIONS),
        # }).sort("count", descending=True))
        
        return Dimension(
            label=emotion,
            labels=emotions,
            pca=pca,
            reg=reg,
            metrics={
                "positive": np.mean(positive_result_arr),
                "negative": np.mean(negative_result_arr),
                "accuracy": metrics.accuracy_score(y_true, y_pred),
                "precision": metrics.precision_score(y_true, y_pred),
                "recall": metrics.recall_score(y_true, y_pred),
                "f1": metrics.f1_score(y_true, y_pred),
                # "roc_auc": metrics.roc_auc_score(y_true, y_pred),
            }
        )

    def train_dimensions(dataset_df: pl.DataFrame, num_dims: int) -> list[Dimension]:
        with st.spinner(text="Training dimensions..."):
            return [train_dimension(emotion, slice(emotion, dataset_df), empty(dataset_df), num_dims) 
                    for emotion in EMOTIONS if emotion != "neutral"]

    def train_dimensions_ekman(dataset_df: pl.DataFrame, num_dims: int) -> list[Dimension]:
        with st.spinner(text="Training dimensions..."):
            # return [train_dimension(emotion, slice(emotion, dataset_df), empty(dataset_df)) 
            #         for emotion in EKMAN_MAP.keys() if emotion != "neutral"]
            return [
                train_dimension(("joy", "sadness"), slice("joy", dataset_df), slice("sadness", dataset_df), num_dims),
                train_dimension(("anger", "fear"), slice("anger", dataset_df), slice("fear", dataset_df), num_dims),
                train_dimension(("surprise", "disgust"), slice("surprise", dataset_df), slice("disgust", dataset_df), num_dims),
            ]

    
    def eval_logistic(emo_df: pl.DataFrame, emotions: list[DimLabel], num_dims: int) -> None:
        def linear_predict(emotion: DimLabel, X: np.ndarray, y: np.ndarray, acc: pl.DataFrame) -> tuple[pl.DataFrame, linear_model.LogisticRegression]:
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

            reg = linear_model.LogisticRegression()
            reg.fit(X_train, y_train)
            y_pred = reg.predict(X_test)

            df = pl.DataFrame({
                "emotion": [emotion.label],
                "accuracy": [metrics.accuracy_score(y_test, y_pred)],
                "precision": [metrics.precision_score(y_test, y_pred)],
                "recall": [metrics.recall_score(y_test, y_pred)],
                "f1": [metrics.f1_score(y_test, y_pred)],
            })
            return acc.vstack(df), reg

        emopoint_result_df = pl.DataFrame({
            "emotion": pl.Series(values=[], dtype=pl.String), 
            "accuracy": pl.Series(values=[], dtype=pl.Float64), 
            "precision": pl.Series(values=[], dtype=pl.Float64), 
            "recall": pl.Series(values=[], dtype=pl.Float64), 
            "f1": pl.Series(values=[], dtype=pl.Float64), 
        })
        orig_result_df = emopoint_result_df.clone()

        for emotion in emotions:
            
            cohort_df = emo_df.select(
                pl.col("embedding").cast(pl.Array(pl.Float32, (num_dims,))).alias("embedding"), 
                pl.col("emopoint"),

                pl.when(pl.col(emotion.positive) == 1).then(1)
                .when(pl.col(emotion.negative) == 1).then(0)
                .otherwise(None)
                .alias("y")
            ).filter(pl.col("y").is_not_null())

            ####################################
            ## By emopoint
            ####################################
            X = cohort_df["emopoint"].to_numpy().reshape(-1, 3)
            y = cohort_df["y"].to_numpy()
            
            emopoint_result_df, reg = linear_predict(emotion, X, y, emopoint_result_df)

            fig = go.Figure(data=[go.Scatter3d(
                x=X[:, 0], y=X[:, 1], z=X[:, 2],
                mode='markers',
                marker=dict(
                    color=cohort_df.select(
                        pl.when(pl.col("y") == 1).then(pl.lit('#ff7f0e'))
                        .otherwise(pl.lit('#1f77b4'))
                        .alias("color")
                    )["color"].to_numpy(), 
                    size=1,
                    opacity=0.5,
                ),
            )])
            # # Create a meshgrid for the decision boundary
            # NUM_TICKS = 30
            # xx, yy = np.meshgrid(np.linspace(X.min(), X.max(), 30), np.linspace(X.min(), X.max(), NUM_TICKS))
            # a, b, c = reg.coef_[0]
            # d = reg.intercept_[0]
            # zz = ((-a * xx) - (b * yy) - d) / c
            # iz_min, iz_max = np.where(zz < X.min(), zz, -np.inf).argmax(), np.where(zz > X.max(), zz, np.inf).argmin()
            # print("iz_min", iz_min, (iz_min % NUM_TICKS, iz_min // NUM_TICKS), zz.shape)
            # print("iz_max", iz_max, (iz_max % NUM_TICKS, iz_max // NUM_TICKS), zz.shape)
            # fig.add_trace(go.Surface(
            #     x=xx, y=yy, z=zz,
            #     colorscale=[[0, 'rgba(0, 0, 0, 0.4)'], [1, 'rgba(0, 0, 0, 0.4)']],
            #     showscale=False,
            # ))
            fig.update_layout(
                title=emotion.label,
                scene=dict(
                    xaxis=dict(title=emotions[0].label),
                    yaxis=dict(title=emotions[1].label),
                    zaxis=dict(title=emotions[2].label),
                ),
            )
            st.plotly_chart(fig)
            fig.write_html(f"./output/emopoint_{emotion.label}.html") 

            ####################################
            ## By original embedding space
            ####################################
            X = cohort_df["embedding"].to_numpy().reshape(-1, num_dims)
            y = cohort_df["y"].to_numpy()
            
            orig_result_df, reg = linear_predict(emotion, X, y, orig_result_df)
            
        st.subheader("Emopoint Performance")
        st.table(emopoint_result_df)
        st.subheader(f"{num_dims}-dim Performance")
        st.table(orig_result_df)


    if model.path.exists():
        st.html('<span style="color: green">File Exists</span>')
    else:
        st.html('<span style="color: red">File Not Exists</span>')

    if st.button("Generate Embeddings"):
        dataset_df = model.generate_df(raw_df)
    elif model.path.exists():
        emotions = [e for e in EKMAN_MAP.keys()]
        dataset_df = load_embeddings(model.path)
        if not all(e in dataset_df.columns for e in emotions):
            dataset_df = dataset_df.join(
                raw_df.select(
                    "id",
                    *emotions,
                ),
                ["id"],
                "inner",
            )
    else:
        return 

    # if st.button("Train Model"):
    st.subheader("Single Dimension Analysis")
    dims = train_dimensions_ekman(dataset_df, model.num_dimensions)
    st.table(
        pl.DataFrame({
            "emotion": [dim.label for dim in dims],
            # "positive": [dim.metrics["positive"] for dim in dims],
            # "negative": [dim.metrics["negative"] for dim in dims],
            "accuracy": [dim.metrics["accuracy"] for dim in dims],
            "precision": [dim.metrics["precision"] for dim in dims],
            "recall": [dim.metrics["recall"] for dim in dims],
            "f1": [dim.metrics["f1"] for dim in dims],
            # "roc_auc": [dim.metrics["roc_auc"] for dim in dims],
        })
    )

    emo_model = EmoModel(
        weights=np.array([dim.pca.components_ for dim in dims]),
        dims=[DimLabel(*dim.ordered_labels) for dim in dims],
        num_emb_dims=model.num_dimensions,
    )
    model_path = pathlib.Path(f"output/model-{model.label}.json")
    model_path.write_text(emo_model.to_json())

    st.subheader("3D space")
    with st.spinner(text="Predicting..."):
        dataset_df = dataset_df.with_columns(pl.col("embedding").cast(pl.Array(pl.Float32, (model.num_dimensions,))))
        result = emo_model.emb_to_emo(dataset_df["embedding"].to_numpy())
        emo_df = dataset_df.hstack([pl.Series(values=result.T, name="emopoint")])
        eval_logistic(emo_df, emo_model.dims, num_dims=model.num_dimensions)


    emotion_colors = {
        "anger": "#ff7f0e",
        "disgust": "#2ca02c",
        "fear": "#d62728",
        "joy": "#9467bd",
        "sadness": "#8c564b",
        "surprise": "#e377c2",
    }
    plot_df = emo_df.select(
            pl.when(pl.col("anger") == 1).then(pl.lit(emotion_colors["anger"]))
            .when(pl.col("anger") == 1).then(pl.lit(emotion_colors["anger"]))
            .otherwise(pl.lit('#000000'))
            .alias("color")
        ) 
    X = plot_df["embedding"].to_numpy().reshape(-1, 3)
    fig = go.Figure(data=[go.Scatter3d(
        x=X[:, 0], y=X[:, 1], z=X[:, 2],
        mode='markers',
        marker=dict(
            color=emo_df.select(
                pl.when(pl.col("anger") == 1).then(pl.lit(emotion_colors["anger"]))
                .when(pl.col("anger") == 1).then(pl.lit(emotion_colors["anger"]))
                .otherwise(pl.lit('#000000'))
                .alias("color")
            )["color"].to_numpy(), 
            size=1,
            opacity=1,
        ),
    )])

