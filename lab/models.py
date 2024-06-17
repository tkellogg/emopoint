import dataclasses
import os
import pathlib

import openai
import polars as pl
import streamlit as st
import tiktoken as tk


@dataclasses.dataclass
class Model:
    id: str
    label: str
    opts: dict = dataclasses.field(default_factory=dict)
    max_tokens: int = 8191
    docs: str = ""

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


MODELS = [
    Model("text-embedding-ada-002", "ada-2", 
          docs="Emopoint extractor tuned for OpenAI's text-embedding-ada-002"),
    Model("text-embedding-3-small", "ada-3-small",
          docs="Emopoint extractor tuned for OpenAI's text-embedding-3-small with 1536 dimensions"),
    Model("text-embedding-3-large", "ada-3-large-256d", opts={"dimensions": 256},
          docs="Emopoint extractor tuned for OpenAI's text-embedding-3-large with 256 dimensions"),
    Model("text-embedding-3-large", "ada-3-large-3072d", opts={"dimensions": 3072},
          docs="Emopoint extractor tuned for OpenAI's text-embedding-3-large with 3072 dimensions"),
]