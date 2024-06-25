import dataclasses
import pathlib
import re
from unittest import mock
from matplotlib import pyplot as plt, dates as mdates
import numpy as np
import polars as pl
import streamlit as st

from models import Model
from emopoint import MODELS, EmoModel
import wikipediaapi


@st.cache_data
def read_truth_social_df() -> pl.DataFrame:
    return pl.read_csv(
        "data/truth_social/truths.tsv", 
        separator="\t",
        schema={
            "id": pl.UInt32(),
            "timestamp": pl.String(),
            "time_scraped": pl.String(),
            "is_retruth": pl.String(),
            "is_reply": pl.String(),
            "author": pl.UInt32(),
            "like_count": pl.UInt64(),
            "retruth_count": pl.UInt64(),
            "reply_count": pl.UInt64(),
            "text": pl.String(),
            "external_id": pl.String(),
            "url": pl.String(),
            "truth_retruthed": pl.String(),
        },
        truncate_ragged_lines=True,
    )

@st.cache_data
def get_truth_social_embeddings(_truths_df: pl.DataFrame, model: Model, gen_embeddings) -> pl.DataFrame:
    path = pathlib.Path(f"data/truth_social/embeddings-{model.label}.parquet")
    if path.exists():
        return pl.read_parquet(path)
    elif not gen_embeddings:
        st.html('<p style="color: red">Embeddings Not Present</p>')
        st.stop()

    model = dataclasses.replace(model)
    return model.generate_df(_truths_df, path=path)


@st.cache_data
def get_wikipedia_texts():
    wiki = wikipediaapi.Wikipedia(user_agent="https://github.com/tkellogg/emopoint")
    def download(article: str) -> list[str]:
        text = wiki.page(article).text
        return re.split(r'\n\s*\n', text)
    def flatten(texts: list[list[str]]) -> list[str]:
        return [text for paragraph in texts for text in paragraph]
    paragraphs = flatten([
        download("Python (programming language)"),
        download("ISO 8601"),
        download("Metric System"),
        download("HTTP"),
        download("Periodic Table"),
        download("Boolean Algebra"),
        download("Geologic time scale"),
        download("Prime number"),
        download("Internet protocol suite"),
        download("Archimedes' principle"),
        download("Newton's laws of motion"),
        download("Ohm's law"),
        download("International System of Units"),
        download("DNA"),
        download("Plate tectonics"),
        download("Morse code"),
        download("Fibonacci sequence"),
        download("Avogadro constant"),
        download("Hertz"),
        download("IPv4"),
        download("Binary number"),
        download("Electromagnetic spectrum"),
        download("Euler's formula"),
        download("Gross domestic product"),
        download("Newton (unit)"),
        download("HTML"),
        download("Proton"),
        download("ASCII"),
        download("Entropy"),
        download("Fourier transform"),
        download("Block (periodic table)"),
        download("Longitude"),
        download("Scientific method"),
    ])
    return paragraphs

@st.cache_data
def get_wikipedia_embeddings(model: Model, gen_embeddings) -> pl.DataFrame:
    path = pathlib.Path(f"data/wikipedia/embeddings-{model.label}.parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return pl.read_parquet(path)
    elif not gen_embeddings:
        st.html('<p style="color: red">Wikipedia Embeddings Not Present</p>')
        st.stop()

    model = dataclasses.replace(model)
    df = pl.DataFrame({"text": get_wikipedia_texts()})
    return model.generate_df(df, path=path)


def main(model: Model):
    truth_social_path = pathlib.Path(f"data/truth_social/embeddings-{model.label}.parquet")
    wikipedia_path = pathlib.Path(f"data/wikipedia/embeddings-{model.label}.parquet")

    emo_model = next(m for m in MODELS if m.label == model.label)
    assert isinstance(emo_model, EmoModel)

    truths_df = read_truth_social_df()
    truth_social_df = get_truth_social_embeddings(
        truths_df
        .select(
            "id",
            pl.col("timestamp").str.to_datetime(strict=False), 
            "text",
            "author",
        ).filter(pl.col("text").is_not_null())
        # .filter(pl.col("text").str.len_chars() < 200)
        .filter(pl.col("author") == pl.lit(37))
        ,
        model,
        st.button("Generate Truth Social Embeddings") if not truth_social_path.exists() else False,
    )

    # with st.spinner("Calculating Truth Social Posts..."):
    #     truth_social_emb_arr = truth_social_df["embedding"].cast(pl.Array(pl.Float32, model.num_dimensions)).to_numpy()
    #     print("TS orig shape:", truth_social_emb_arr.shape)
    #     truth_social_mags = emo_model.emotional_magnitude(truth_social_emb_arr)
    #     print("TS mag shape:", truth_social_mags.shape)

    wikipedia_df = get_wikipedia_embeddings(
        model,
        st.button("Generate Wikipedia Embeddings") if not wikipedia_path.exists() else False,
    )
    with st.spinner("Calculating Wikipedia Posts..."):
        wikipedia_emb_arr = wikipedia_df["embedding"].cast(pl.Array(pl.Float32, model.num_dimensions)).to_numpy()
        wikipedia_mags = emo_model.emotional_magnitude(wikipedia_emb_arr)

    # fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    # ax1.hist(
    #     wikipedia_mags,
    #     bins=40,
    #     alpha=0.3,
    #     label='Wikipedia',
    #     color='r'
    # )
    # ax2.hist(
    #     truth_social_mags,
    #     bins=40,
    #     alpha=0.7,
    #     label='Truth Social'
    # )
    # ax2.set_title("Amount of Emotion, Trump vs Wikipedia")
    # ax2.set_xlabel("Portion of Emotion")
    # ax2.set_ylabel("Count (Truth Social)")
    # ax1.set_ylabel("Count (Wikipedia)")
    # ax2.legend(loc='upper left')
    # ax1.legend(loc='upper right')

    # st.pyplot(fig)



    wiki_emopoints = emo_model.emb_to_emo(wikipedia_emb_arr)
    truth_social_emb_arr = truth_social_df["embedding"].cast(pl.Array(pl.Float32, model.num_dimensions)).to_numpy()
    ts_emopoints = emo_model.emb_to_emo(truth_social_emb_arr)

    def plot_hist(dim: int):
        wiki_hist, wiki_bins = np.histogram(wiki_emopoints[:, dim], bins=40, density=True)
        ts_hist, ts_bins = np.histogram(ts_emopoints[:, dim], bins=40, density=True)
        wiki_hist_percent = (wiki_hist / np.sum(wiki_hist)) * 100
        ts_hist_percent = (ts_hist / np.sum(ts_hist)) * 100

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.bar(
            wiki_bins[:-1],
            wiki_hist_percent,
            width=np.diff(wiki_bins)[0],
            alpha=0.3,
            label="Wikipedia",
            color='b'
        )
        ax2.bar(
            ts_bins[:-1],
            ts_hist_percent,
            width=np.diff(ts_bins)[0],
            alpha=0.3,
            label="Trump",
            color='r'
        )
        ax1.set_title(emo_model.dims[dim].label)
        ax1.set_xlabel(f"{emo_model.dims[dim].negative} ← Intensity → {emo_model.dims[dim].positive}")
        ax2.set_ylabel("Trump (% of Truth Social posts)")
        ax1.set_ylabel("Wikipedia (% of sample paragraphs)")
        # ax1.axvline(x=np.percentile(wiki_emopoints[:, dim], 99), color='r', linestyle='--')
        # ax1.axvline(x=np.percentile(wiki_emopoints[:, dim], 1), color='r', linestyle='--')
        ax2.legend(loc='upper left')
        ax1.legend(loc='upper right')
        ax1.set_ylim(0, max(wiki_hist_percent.max(), ts_hist_percent.max()) + 2)
        ax2.set_ylim(0, max(wiki_hist_percent.max(), ts_hist_percent.max()) + 2)
        ax1.set_xlim(-0.4, 0.4)
        ax2.set_xlim(-0.4, 0.4)
        st.pyplot(fig)

    plot_hist(0)
    plot_hist(1)
    plot_hist(2)

    def plot_intensity():
        wiki_mag = emo_model.emotional_magnitude(wikipedia_emb_arr)
        wiki_hist, wiki_bins = np.histogram(wiki_mag, bins=40, density=True)
        ts_mag = emo_model.emotional_magnitude(truth_social_emb_arr)
        ts_hist, ts_bins = np.histogram(ts_mag, bins=40, density=True)
        wiki_hist_percent = (wiki_hist / np.sum(wiki_hist)) * 100
        ts_hist_percent = (ts_hist / np.sum(ts_hist)) * 100

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        ax1.bar(
            wiki_bins[:-1],
            wiki_hist_percent,
            width=np.diff(wiki_bins)[0],
            alpha=0.3,
            label="Wikipedia",
            color='b'
        )
        ax2.bar(
            ts_bins[:-1],
            ts_hist_percent,
            width=np.diff(ts_bins)[0],
            alpha=0.3,
            label="Trump",
            color='r'
        )
        st.pyplot(fig)

    plot_intensity()

    def plot_timeline(dim: int):
        # line graph of Trump magnitude by date
        mags_df = (
            pl.DataFrame({
                "timestamp": truth_social_df["timestamp"],
                "magnitude": ts_emopoints[:, dim].reshape(-1),
            })
            .sort("timestamp")
            .group_by(pl.col("timestamp").dt.date()).agg(
                pl.col("magnitude").quantile(0.1).alias(emo_model.dims[dim].negative),
                pl.col("magnitude").quantile(0.9).alias(emo_model.dims[dim].positive),
            )
        )
        fig, ax = plt.subplots()
        ax.plot(
            mags_df["timestamp"], 
            mags_df[emo_model.dims[dim].positive], 
            label="90th percentile",
        )
        ax.plot(
            mags_df["timestamp"], 
            mags_df[emo_model.dims[dim].negative], 
            label="10th percentile",
        )
        ax.set_title(f"{emo_model.dims[dim].label} throughout 2022")
        ax.set_ylabel(f"← {emo_model.dims[dim].negative} ← Intensity → {emo_model.dims[dim].positive} →")
        ax.set_xlabel("Date of Trump's Posts")
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.legend()
        st.pyplot(fig)
    
    plot_timeline(0)
    plot_timeline(1)
    plot_timeline(2)

    st.table(
        truth_social_df.sort("timestamp", descending=True)
        .select("timestamp", "text")
        .limit(20)
    )

    ts_w_mags_df = truth_social_df
    columns = {"emotion": [], "count": [], "percent": []}
    for iDim, dim in enumerate(emo_model.dims):
        ts_w_mags_df = ts_w_mags_df.with_columns([
            pl.int_range(0, len(truth_social_df), 1)
            .map(lambda idx: ts_emopoints[idx, iDim])
            .alias(dim.label)
        ])

        pos_thresh = np.percentile(wiki_emopoints[:, iDim], 99)
        num_ts_pos = len(ts_emopoints[ts_emopoints[:, iDim] > pos_thresh])
        columns["emotion"].append(dim.positive)
        columns["count"].append(num_ts_pos)
        columns["percent"].append(num_ts_pos / len(ts_emopoints))
        st.subheader(dim.positive)
        st.table(
            ts_w_mags_df.filter(pl.col(dim.label) > pos_thresh)
            .sample(10)
            .select("text")
        )

        neg_thresh = np.percentile(wiki_emopoints[:, iDim], 1)
        num_ts_neg = len(ts_emopoints[ts_emopoints[:, iDim] < neg_thresh])
        columns["emotion"].append(dim.negative)
        columns["count"].append(num_ts_neg)
        columns["percent"].append(num_ts_neg / len(ts_emopoints))
        st.subheader(dim.negative)
        st.table(
            ts_w_mags_df.filter(pl.col(dim.label) < neg_thresh)
            .sample(10)
            .select("text")
        )



    st.table(pl.DataFrame(columns))