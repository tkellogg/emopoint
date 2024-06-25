import dataclasses
import functools
import os
import pathlib

import dotenv
import numpy as np
import openai
import polars as pl
import streamlit as st
import tiktoken as tk


dotenv.load_dotenv()

@dataclasses.dataclass
class _Buffer:
    path_base: pathlib.Path
    text_buf: list[str]
    emb_buf: list[list[float]]
    max_size: int = 100
    predicates: pl.Series | None = None
    index: int = 0

    def add(self, text: list[str], emb: list[list[float]]):
        self.text_buf.extend(text)
        self.emb_buf.extend(emb)

        if len(self.text_buf) > self.max_size:
            self.flush()

    def flush(self):
        df = pl.DataFrame({
            "text": self.text_buf,
            "embeddings": self.emb_buf,
        })
        path = f"{self.path_base}.{self.index}"
        df.write_parquet(path)
        self.text_buf.clear()
        self.emb_buf.clear()

    def get_embeddings(self) -> pl.Series:
        import glob
        if len(self.emb_buf) > 0:
            self.flush()

        # read and sort
        buf = []
        for f in glob.glob(f"{self.path_base}.*"):
            df = pl.read_parquet(f)
            index = int(str(f).split(".")[-1])
            buf.append((index, df["embeddings"]))
        
        # construct sorted series
        data = functools.reduce(
            lambda acc, b: acc.append(b),
            (b for _, b in sorted(buf, key=lambda x: x[0])),
        )
        return pl.Series(data, dtype=pl.List(pl.Float64()))

    def init_texts(self, texts: pl.Series):
        try:
            all = pl.read_parquet(f"{self.path_base}.*")
        except FileNotFoundError:
            return texts.map_elements(lambda x: False, pl.Boolean())
        texts_df = pl.DataFrame({"text": texts})
        self.predicates = texts_df.select(
            pl.col("text").is_in(all["text"]).alias("pred")
        )["pred"]
        print(self.predicates.dtype, self.predicates.shape)

    def has_index(self, index: int) -> bool:
        self.index = index
        return self.predicates is not None and self.predicates[index]



@dataclasses.dataclass
class Model:
    id: str
    label: str
    opts: dict = dataclasses.field(default_factory=dict)
    max_tokens: int = 8191
    docs: str = ""
    base_url: str | None = None

    @property
    def path(self) -> pathlib.Path:
        return pathlib.Path(f"data/models/{self.label}.parquet")

    @property
    def num_dimensions(self) -> int:
        return self.opts.get("dimensions", 1536)

    def generate_df(self, raw_df: pl.DataFrame, path: pathlib.Path | None = None) -> pl.DataFrame:
        """
        Generate embeddings for all texts for this model.
        """
        try:
            enc = tk.encoding_for_model(self.id)
        except KeyError:
            enc = tk.encoding_for_model("text-embedding-3-small")

        oai = openai.OpenAI(api_key=os.environ["OPENAI_KEY"], base_url=self.base_url)
        
        def embed(texts: pl.Series) -> pl.Series:
            buf = _Buffer(path_base=path or self.path, text_buf=[], emb_buf=[])
            buf.init_texts(texts)

            input_buf = []
            output_buf = []
            num_toks = 0
            skip_count = 0
            for i, text in enumerate(texts):
                if buf.has_index(i):
                    skip_count += 1
                    if skip_count % 100 == 0:
                        print(".", end="")
                    continue
                if text is None:
                    raise ValueError("Filter for nulls first")
                tok_buf = enc.encode(text)
                toks = len(tok_buf)
                if num_toks + toks > self.max_tokens:
                    print("Flushing", len(input_buf), "texts")
                    resp = oai.embeddings.create(input=input_buf, model=self.id, **self.opts)
                    output_buf.extend([d.embedding for d in resp.data])
                    buf.add(input_buf, [d.embedding for d in resp.data])
                    num_toks = 0
                    input_buf.clear()
                if toks > self.max_tokens:
                    print("Flushing 1 text (big one)")
                    shortened = enc.decode([t1 for t1 in tok_buf][:self.max_tokens - 2])
                    resp = oai.embeddings.create(input=shortened, model=self.id, **self.opts)
                    output_buf.extend([d.embedding for d in resp.data])
                    buf.add(input_buf, [d.embedding for d in resp.data])
                    continue
                num_toks += toks
                input_buf.append(text)

            if len(input_buf) > 0:
                print("Flushing", len(input_buf), "texts (final)")
                resp = oai.embeddings.create(input=input_buf, model=self.id, **self.opts)
                output_buf.extend([d.embedding for d in resp.data])
                buf.add(input_buf, [d.embedding for d in resp.data])

            # return pl.Series(output_buf)
            return buf.get_embeddings()

        with st.spinner(text="Generating embeddings..."):
            dataset_df = raw_df.with_columns(
                pl.col("text").map_batches(embed, return_dtype=pl.List(pl.Float64)).alias("embedding")
            )
            self.path.parent.mkdir(parents=True, exist_ok=True)
            print(path or self.path)
            dataset_df.write_parquet(path, compression="snappy")
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

    Model("clap", "clap", opts={"dimensions": 512},
          docs="Emopoint extractor tuned for CLAP, a multi-modal model that supports audio and text",
          base_url="http://localhost:4001/v1"),
]