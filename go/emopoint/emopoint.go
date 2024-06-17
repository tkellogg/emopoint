// A library for extracting emotion from embeddings.
//
// An emopoint is a simplified embedding with interpretable dimensions:
//  1. joy vs sadness
//  2. anger vs fear
//  3. disgust vs surprise
//
// This library enables you to either convert from embeddings to emopoints, or to
// stay within embedding space but remove emotional information.
package emopoint

import "fmt"

// An axis label
type DimLabel struct {
	// Numbers less than zero refer to this
	Negative string
	// Numbers greater than zero refer to this
	Positive string
}

// A model that's been trained to extract emotional information from a specific
// embedding model. Instances of this struct are declared in this package and
// correspond to specific out-of-the-box embedding models with various settings.
type EmoModel struct {
	Weights                [][]float32
	Dims                   []DimLabel
	numEmbeddingDimensions int
}

// Convert an embedding to an emopoint. Emopoints are a [3]float32 representing
// only emotional information.
func (model *EmoModel) EmbeddingToEmopoint(embedding []float32) ([]float32, error) {
	if len(embedding) != model.numEmbeddingDimensions {
		return nil, &SizeError{
			Code:    1,
			Message: fmt.Sprintf("Expected embedding to be of length %d but was %d", model.numEmbeddingDimensions, len(embedding)),
		}
	}
	res := make([]float32, 3)
	for iWeight, weightVec := range model.Weights {
		for iEmb, emb := range embedding {
			res[iWeight] += emb * weightVec[iEmb]
		}
	}
	return res, nil
}

// Convert a set of embeddings to emopoints. Emopoints are a [3]float32 representing
// only emotional information. This returns a list of emopoints.
func (model *EmoModel) EmbeddingToEmopointMulti(embeddings [][]float32) ([][]float32, error) {
	res := make([][]float32, len(embeddings))
	for iRow, embedding := range embeddings {
		if len(embedding) != model.numEmbeddingDimensions {
			return nil, &SizeError{
				Code:    1,
				Message: fmt.Sprintf("Expected embedding to be of length %d but was %d at index %d", model.numEmbeddingDimensions, len(embedding), iRow),
			}
		}

		res[iRow] = make([]float32, 3)
		for iWeight, weightVec := range model.Weights {
			for iEmb, emb := range embedding {
				res[iRow][iWeight] += emb * weightVec[iEmb]
			}
		}
	}
	return res, nil
}

// Take an embedding and remove emotional information. The returned list is of the
// same dimensions as the input.
func (model *EmoModel) RemoveEmotion(embedding []float32) ([]float32, error) {
	if len(embedding) != model.numEmbeddingDimensions {
		return nil, &SizeError{
			Code:    1,
			Message: fmt.Sprintf("Expected embedding to be of length %d but was %d", model.numEmbeddingDimensions, len(embedding)),
		}
	}
	res := make([]float32, model.numEmbeddingDimensions)
	for iEmb, emb := range embedding {
		var total float32 = 0.0
		for _, weightVec := range model.Weights {
			total += emb * weightVec[iEmb]
		}
		res[iEmb] = emb - total
	}
	return res, nil
}

// Take a set of embeddings and remove emotional information. The returned list is of the
// same dimensions as the input.
func (model *EmoModel) RemoveEmotionMulti(embeddings [][]float32) ([][]float32, error) {
	res := make([][]float32, len(embeddings))
	for iRow, embedding := range embeddings {
		if len(embedding) != model.numEmbeddingDimensions {
			return nil, &SizeError{
				Code:    1,
				Message: fmt.Sprintf("Expected embedding to be of length %d but was %d", model.numEmbeddingDimensions, len(embedding)),
			}
		}
		res[iRow] = make([]float32, len(embedding))
		for iEmb, emb := range embedding {
			var total float32 = 0.0
			for _, weightVec := range model.Weights {
				total += emb * weightVec[iEmb]
			}
			res[iRow][iEmb] = emb - total
		}
	}
	return res, nil
}

type SizeError struct {
	Code    int
	Message string
}

func (e *SizeError) Error() string {
	return fmt.Sprintf("SizeError %d: %s", e.Code, e.Message)
}
