package emopoint_test

import (
	"math"
	"testing"

	emo "github.com/tkellogg/emopoint/go/emopoint"
)

var model2d emo.EmoModel = emo.NewEmoModel(
	[][]float32{
		{0.1, 0.1},
		{0.0, 0.0},
		{0.1, 0.0},
	},
	[]emo.DimLabel{
		{Positive: "good", Negative: "bad"},
		{Positive: "totally", Negative: "kinda"},
		{Positive: "historical", Negative: "fad"},
	},
	2,
)

func TestEmbeddingToEmopoint(t *testing.T) {
	input := []float32{1.0, 1.0}
	res, err := model2d.EmbeddingToEmopoint(input)
	if err != nil {
		t.Error(err.Error())
		return
	}
	if len(res) != 3 {
		t.Errorf("Expected res to have 3 items but had %d", len(res))
		return
	}
	expected := []float32{0.2, 0.0, 0.1}
	if !arraysEqual(res, expected) {
		t.Errorf("Array value wrong\nExpected: %v\nActual:   %v", res, expected)
	}
}

func TestEmbeddingToEmopointMulti(t *testing.T) {
	input := [][]float32{
		{1.0, 1.0},
		{0.1, 0.1},
	}
	res, err := model2d.EmbeddingToEmopointMulti(input)
	if err != nil {
		t.Error(err.Error())
		return
	}
	if len(res) != 2 {
		t.Errorf("Expected res to have 2 items but had %d", len(res))
		return
	}
	expected := [][]float32{
		{0.2, 0.0, 0.1},
		{0.02, 0.0, 0.01},
	}
	if !arraysEqual2d(res, expected) {
		t.Errorf("Array value wrong\nExpected: %v\nActual:   %v", res, expected)
	}
}

func TestRemoveEmotion(t *testing.T) {
	input := []float32{1.0, 1.0}
	res, err := model2d.RemoveEmotion(input)
	if err != nil {
		t.Error(err.Error())
		return
	}
	if len(res) != 2 {
		t.Errorf("Expected res to have 2 items but had %d", len(res))
		return
	}
	expected := []float32{0.8, 0.9}
	if !arraysEqual(res, expected) {
		t.Errorf("Array value wrong\nExpected: %v\nActual:   %v", res, expected)
	}
}

func TestRemoveEmotionMulti(t *testing.T) {
	input := [][]float32{
		{1.0, 1.0},
		{0.1, 0.1},
	}
	res, err := model2d.RemoveEmotionMulti(input)
	if err != nil {
		t.Error(err.Error())
		return
	}
	if len(res) != 2 {
		t.Errorf("Expected res to have 2 items but had %d", len(res))
		return
	}
	expected := [][]float32{
		{0.8, 0.9},
		{0.08, 0.09},
	}
	if !arraysEqual2d(res, expected) {
		t.Errorf("Array value wrong\nExpected: %v\nActual:   %v", res, expected)
	}
}

func arraysEqual(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	epsilon := 0.00001

	for i := range a {
		// Equality with a float tolerance
		if math.Abs(float64(a[i]-b[i])) > epsilon {
			return false
		}
	}

	return true
}

func arraysEqual2d(a, b [][]float32) bool {
	for i := range a {
		if !arraysEqual(a[i], b[i]) {
			return false
		}
	}
	return true
}
