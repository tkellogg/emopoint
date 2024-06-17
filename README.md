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