Extract emotional information from embeddings.

When working with LLMs, various embedding models capture emotional information 
that might be useful to work with (or without!). 

An emopoint is a simplified embedding with interpretable dimensions:
 1. joy vs sadness
 2. anger vs fear
 3. disgust vs surprise

So, for example OpenAI's `text-embedding-3-small` returns embeddings with 1536
dimensions. This library will convert those into 3 dimensions, losing most
information except for what directly relates to emotion.

This library enables two modes:
 1. Isolate emotion, converting it into 3D emopoint vectors
 2. Remove emotion, stay in original dimensionality

# Install
Install using your language's package manager:

## [JavaScript/TypeScript via NPM](https://www.npmjs.com/package/emopoint)
```bash
npm i emopoint
```

and then use it

```javascript
const { MODELS } = require('emopoint');

console.log(MODELS.ADA_2);
```

## [Python via PyPi](https://pypi.org/project/emopoint/)
```bash
pip install emopoint
```
and then use it

```python
from emopoint import MODELS

embedding = get_embeddings("James was maaaaaad")
emopoint = MODELS.ADA_3_SMALL.emb_to_emo(embedding)
```

## Go
```bash
go get github.com/tkellogg/emopoint/go/emopoint
```

and then use it

```go
import (
	emo "github.com/tkellogg/emopoint/go/emopoint"
)

func main() {
	var embeding []float32 = getEmbeddings("James was maaaaaad")
	var emopoint []float32 = emo.ADA_3_SMALL.EmbeddingToEmopoint(embedding)
}
```


# Functions
All 3 languages have these capabilities:
* Convert embedding to emopoint — Convert an embedding (e.g. 1536 dimensions for `text-embedding-3-small`) to 3-dimensional space,
  called `emopoint` space that represents only emotion and nothing else.
* Remove emotion — Take an embedding and keep it in the same dimensionality, but subtract emotional information

From these operations, there's a lot more you can do:
* Get the portion of emotional information in text — Calculate the magnitude of the embedding (should be always `1.0`) and subtract
  the magnitude of the result of `remove_emotion(embedding)`. The result is a scalar `float` that represents the portion of the
  meaning of the text that was dedicated to emotion, as the embedding model understood it.
* Cluster on emotion — Convert to `emopoint` space and run a K-Means clustering algorithm
* Semantic search on emotion only — Convert to `emopoint` space and store in a vector database. This matches text based only on the
  emotional content, ignoring all factual and subjective information.
* Semantic search without emotion — Same as before, but store the result of `remove_emotion(embedding)`. This removes noise introduced
  by emotion, creating closer matches and potentially enhancing the search accuracy.
* Analytics & visualizations on emotional magnitude — Calculate the magnitudes of emopoints for several texts, e.g. sections of a speech 
  or tweets, and create visualizations on just the magnitude (portion of information dedicated to emotion).
* Analytics & visualizations on emotions — Same as before, but instead of calculating the magnitude, visualize the points in 3D emopoint
  space. Observe how some texts lean toward anger or joy. Analyze how emotions ebb & flow throughout a speech, and contrast that to
  the informational content (maybe use K-Means clustering on original content to classify the content and display those classifications
  as colors in a [3D scatter plot](https://plotly.com/python/3d-scatter-plots/)).