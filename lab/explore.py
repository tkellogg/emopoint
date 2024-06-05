import functools
import polars as pl
import streamlit as st

from emopoint import EMOTIONS


def main():
    raw_df = pl.concat([
        pl.read_csv("data/goemotions_1.csv"),
        pl.read_csv("data/goemotions_2.csv"),
        pl.read_csv("data/goemotions_3.csv"),
    ])

    nonneutral_emotions = [e for e in EMOTIONS if e != "neutral"]

    num_emotions_df = raw_df.select(
        functools.reduce(
            lambda acc, emotion: acc + pl.col(emotion),
            nonneutral_emotions,
            pl.lit(0),
        ).alias("count"),
    )

    st.header("How many emotions per sentence?")
    st.table(
        num_emotions_df
        .select(
            pl.col("count").max().alias("max"),
            pl.col("count").min().alias("min"),
            pl.col("count").mean().alias("mean"),
        )
    )

    st.metric("total count", num_emotions_df.shape[0])
    st.metric("# no emotion", num_emotions_df.filter(pl.col("count") == 0).shape[0])
    st.metric("% no emotion", (num_emotions_df.filter(pl.col("count") == 0).count() * 100) / num_emotions_df.count())

    st.header("Count by emotion")
    st.bar_chart((
        pl.concat([
            (
                raw_df.melt(id_vars=["id"], value_vars=EMOTIONS, variable_name="emotion", value_name="count")
                # .top_k(10, by=["count"])
                .group_by("emotion").agg(pl.col("count").sum().alias("count"))
            ),
            (
                num_emotions_df
                .filter(pl.col("count") == pl.lit(0))
                .select(pl.lit("NOT LABELED").alias("emotion"), pl.len().cast(pl.Int64).alias("count"))
            ),
        ])
    ), x="emotion", y="count")


    st.header("Text Size (tokens a la ada-2)")

    def count_tok(texts: pl.Series) -> int:
        import tiktoken as tk
        enc = tk.encoding_for_model("text-embedding-ada-002")
        # enc = tk.Encoding("text-embedding-ada-002")
        return pl.Series([len(b) for b in enc.encode_batch(texts.to_list())], dtype=pl.Int64)

    @st.cache_data
    def count_tok_df() -> pl.DataFrame:
        return (
            raw_df
            .select(pl.col("text").map_batches(count_tok, return_dtype=pl.Int64).alias("text_size"))
        )

    st.table(
        count_tok_df()
        .select(
            pl.col("text_size").min().alias("min"),
            pl.col("text_size").mean().alias("mean"),
            pl.col("text_size").max().alias("max"),
        )
    )