import dataclasses
import functools
import os
import pathlib
import dotenv
import matplotlib.pyplot as plt
import numpy as np
import openai
import polars as pl
from sklearn import metrics, linear_model
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import streamlit as st
import tiktoken as tk

from emopoint import EMOTIONS


dotenv.load_dotenv()


@dataclasses.dataclass
class Model:
    id: str
    label: str
    opts: dict = dataclasses.field(default_factory=dict)
    max_tokens: int = 8191

    @property
    def path(self) -> pathlib.Path:
        return pathlib.Path(f"data/models/{self.label}.parquet")

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

@dataclasses.dataclass
class Dimension:
    label: str
    pca: PCA
    reg: linear_model.LogisticRegression
    metrics: dict[str, float]

def main():

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

    empty_df = empty(raw_df)

    def slice(emotion: str, dataset_df: pl.DataFrame=raw_df) -> pl.DataFrame:
        return dataset_df.filter(pl.col(emotion) > 0)

    def concat(a: pl.DataFrame, b: pl.DataFrame) -> np.ndarray:
        return pl.concat([a, b])["embedding"].cast(pl.Array(pl.Float32, (1536,))).to_numpy()


    def train_dimension(emotion: str, dataset_df: pl.DataFrame) -> Dimension:
        """
        Calculate the dimension of a given emotion.
        """
        signal_df = slice(emotion, dataset_df)
        assert signal_df.shape[0] <= empty_df.shape[0], f"signal_df.shape[0]={signal_df.shape[0]} > empty_df.shape[0]={empty_df.shape[0]}"
        neutral_df = empty(dataset_df).sample(signal_df.shape[0])
        signal_train_df, signal_test_df, neutral_train_df, neutral_test_df = train_test_split(signal_df, neutral_df, train_size=0.8)
        
        train_arr = concat(signal_train_df, neutral_train_df)

        pca = PCA(n_components=1)
        pca.fit(train_arr)

        signal_result_arr = pca.transform(signal_test_df["embedding"].cast(pl.Array(pl.Float32, (1536,))).to_numpy())
        neutral_result_arr = pca.transform(neutral_test_df["embedding"].cast(pl.Array(pl.Float32, (1536,))).to_numpy())

        fig, ax = plt.subplots()
        ax.hist(signal_result_arr, bins=30, alpha=0.5, label='Signal', edgecolor='black')
        ax.hist(neutral_result_arr, bins=30, alpha=0.5, label='Neutral', edgecolor='black')

        # Adding labels, title, and legend
        ax.set_title(f"Distribution of {emotion}")
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.legend()

        # Display the histogram in Streamlit
        st.pyplot(fig)

        y_true = np.concatenate([np.ones(len(signal_result_arr)), np.zeros(len(neutral_result_arr))])

        reg = linear_model.LogisticRegression()
        reg.fit(np.concatenate([signal_result_arr, neutral_result_arr]), y_true)

        # signal_result_arr = np.array([x < 0 for x in signal_result_arr])
        # neutral_result_arr = np.array([x >= 0 for x in neutral_result_arr])
        signal_result_arr = reg.predict(signal_result_arr)
        neutral_result_arr = reg.predict(neutral_result_arr)
        
        y_pred = np.concatenate([signal_result_arr, neutral_result_arr])
        
        return Dimension(
            label=emotion,
            pca=pca,
            reg=reg,
            metrics={
                "signal": np.mean(signal_result_arr),
                "neutral": np.mean(neutral_result_arr),
                "accuracy": metrics.accuracy_score(y_true, y_pred),
                "precision": metrics.precision_score(y_true, y_pred),
                "recall": metrics.recall_score(y_true, y_pred),
                "f1": metrics.f1_score(y_true, y_pred),
                # "roc_auc": metrics.roc_auc_score(y_true, y_pred),
            }
        )

    def train_dimensions(dataset_df: pl.DataFrame) -> list[Dimension]:
        with st.spinner(text="Training dimensions..."):
            return [train_dimension(emotion, dataset_df) for emotion in EMOTIONS if emotion != "neutral"]


    model = st.selectbox('Model', [
        Model("text-embedding-ada-002", "ada-2"),
        Model("text-embedding-3-small", "ada-3-small"),
        Model("text-embedding-3-large", "ada-3-large (min size)", opts={"dimensions": 1536}),
        Model("text-embedding-3-large", "ada-3-large (max size)", opts={"dimensions": 3072}),
    ], format_func=lambda model: model.label)
    if model.path.exists():
        st.html('<span style="color: green">File Exists</span>')
    else:
        st.html('<span style="color: red">File Not Exists</span>')

    if st.button("Generate Embeddings"):
        dataset_df = model.generate_df(raw_df)
    elif model.path.exists():
        dataset_df = load_embeddings(model.path)
    else:
        return 
    

    # if st.button("Train Model"):
    dims = train_dimensions(dataset_df)
    st.table(
        pl.DataFrame({
            "emotion": [dim.label for dim in dims],
            "signal": [dim.metrics["signal"] for dim in dims],
            "neutral": [dim.metrics["neutral"] for dim in dims],
            "accuracy": [dim.metrics["accuracy"] for dim in dims],
            "precision": [dim.metrics["precision"] for dim in dims],
            "recall": [dim.metrics["recall"] for dim in dims],
            "f1": [dim.metrics["f1"] for dim in dims],
            # "roc_auc": [dim.metrics["roc_auc"] for dim in dims],
        })
    )
