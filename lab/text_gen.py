import collections
import dataclasses
import functools
import itertools
import json
import os

import dotenv
import openai
import polars as pl
import streamlit as st

from emopoint import EKMAN_MAP
import train


dotenv.load_dotenv()


PROMPT = """For the sentences below, rephrase the sentence to show {emotion}. Try to keep the same meaning, but change the emotion. You're allowed some creative liberty. The result must be a JSON object on a single line containing the keys "id" and "text". Respond only in JSON.

Example Input:
eew5j0j) The cat ate his food
eemcysk) Cars are low maintenance

Example output:
{{"id":"eew5j0j","The cat was {emotion} while he ate his food"}}
{{"id":"eemcysk","Cars are low maintenance, and that is {emotion}"}}

Input:
{input}

Output:
"""

def fill_prompt(emotion: str, texts: list["EmoDelta"]) -> str:
    inputs = [f"{text.id}) {text.text}" for text in texts]
    return PROMPT.format(
        emotion=emotion,
        input="\n".join(inputs),
    )

@dataclasses.dataclass
class EmoDelta:
    text: str
    emotion: str
    id: str

    def multiply(self, emotions: list[str]) -> list['EmoDelta']:
        return [dataclasses.replace(self, emotion=e) for e in emotions if e != self.emotion]


def gen_examples(prompt: str, emotion: str) -> list[EmoDelta]:
    oai = openai.OpenAI(api_key=os.environ["OPENAI_KEY"])
    response = oai.chat.completions.create(
        messages=[ dict(role="user", content=prompt) ],
        temperature=0,
        model="gpt-4o",
        # response_format={ "type": "json_object" },
    )

    result = []
    for line in response.choices[0].message.content.splitlines():
        try:
            data = json.loads(line)
        except:
            if line.startswith("```"):
                continue
            print(f"Invalid JSON: {line}")
            continue
        
        result.append(EmoDelta(data["text"], emotion, data["id"]))
    return result


def multiply_text(orig: list[EmoDelta]) -> list[EmoDelta]:
    by_emotion: dict[str, list[EmoDelta]] = collections.defaultdict(list)
    for emo in orig:
        multiple = emo.multiply(EKMAN_MAP.keys())
        for e in multiple:
            by_emotion[e.emotion].append(e)

    output = []
    for emotion, examples in by_emotion.items():
        print(f"Filling prompt with {len(examples)} input items; {len(set(map(lambda x: x.id, examples)))}")
        prompt = fill_prompt(emotion, examples)
        output.extend(gen_examples(prompt, emotion))

    return output

def any_of(emotion) -> pl.Expr:
    return functools.reduce(
        lambda acc, sub: acc | pl.col(sub) == pl.lit(1),
        EKMAN_MAP[emotion],
        pl.lit(False),
    )

def generate_samples(sample=10):
    raw_df = pl.concat([
        pl.read_csv("data/goemotions_1.csv"),
        pl.read_csv("data/goemotions_2.csv"),
        pl.read_csv("data/goemotions_3.csv"),
    ])

    raw_df = raw_df.with_columns([
        pl.when(any_of("anger")).then(pl.lit("anger"))
        .when(any_of("disgust")).then(pl.lit("disgust"))
        .when(any_of("fear")).then(pl.lit("fear"))
        .when(any_of("joy")).then(pl.lit("joy"))
        .when(any_of("sadness")).then(pl.lit("sadness"))
        .when(any_of("surprise")).then(pl.lit("surprise"))
        .otherwise(pl.lit("neutral"))
        .alias("emotion")
    ])

    emotions = list(EKMAN_MAP.keys())
    sampled_df = pl.concat([
        raw_df.filter(any_of(emotion)).sample(sample)
        for emotion in emotions
    ])
    print(f"{len(sampled_df)} original rows")

    objects = [EmoDelta(row["text"], row["emotion"], row["id"]) 
               for row in sampled_df.rows(named=True)]
    result: list[EmoDelta] = []
    for page in batched(objects, n=40):
        print(f"batched; n={len(page)}, first={page[0].text}")
        result.extend(multiply_text(page))
    print(f"expansion done; len={len(result)}") 

    result_df = pl.DataFrame({
        "emotion": [r.emotion for r in result],
        "text": [r.text for r in result],
        "id": [r.id for r in result],
        "orig": ["" for _ in enumerate(result)],
        "seq": [1 for _, _ in enumerate(result)],
    })

    return (
        result_df
        .extend(sampled_df.select(["emotion", "text", "id", pl.lit("Original").alias("orig"), pl.lit(0, pl.Int64).alias("seq")]))
        .sort(["emotion", "seq"], descending=False)
        .drop("seq")
    )


def batched(iterable, n=1):
    cur = []
    for item in iterable:
        cur.append(item)
        if len(cur) > n:
            yield cur
            cur = []
    if len(cur) > 0:
        yield cur


def main(model: train.Model):
    # result_df = generate_samples(2)
    result_df = pl.read_csv("data/expanded_emotions.csv")

    embeddings_df = model.generate_df(result_df)
    embeddings_df.write_parquet(f"data/expanded_emotions-{model.label}.parquet")

    # result_df.write_csv("data/expanded_emotions.csv", include_header=True)
    # st.table(
    #     result_df.sort(["id", "orig"])
    # )

    st.table(
        pl.read_parquet(f"data/expanded_emotions-{model.label}.parquet")
        .group_by("id").agg(pl.implode("emotion"))
    )
