import { DimLabel, EmoModel } from "./lib";

const model2d = new EmoModel(
    [
        new Float32Array([0.1, 0.1]),
        new Float32Array([0.0, 0.0]),
        new Float32Array([0.1, 0.0]),
    ],
    [
        new DimLabel("bad", "good"),
        new DimLabel("kinda", "totally"),
        new DimLabel("fad", "historical"),
    ],
    2
)

describe("EmoPoint", () => {
    describe("embeddingToEmoPoint", () => {
        it("computes into 3D space", () => {
            const input = new Float32Array([1.0, 1.0]);
            const output = model2d.embeddingToEmopoint(input);
            const expected = new Float32Array([0.2, 0.0, 0.1]);
            arraysEqual(output, expected);
        });
    });

    describe("embeddingToEmoPointMulti", () => {
        it("computes into 3D space", () => {
            const input = [
                new Float32Array([1.0, 1.0]),
                new Float32Array([0.1, 0.1]),
            ];
            const output = model2d.embeddingToEmopointMulti(input);
            const expected = [
                new Float32Array([0.2, 0.0, 0.1]),
                new Float32Array([0.02, 0.0, 0.01]),
            ]
            arraysEqual2d(output, expected);
        });
    });

    describe("removeEmotion", () => {
        it("subtracts emotional value from embedding", () => {
            const input = new Float32Array([1.0, 1.0]);
            const output = model2d.removeEmotion(input);
            const expected = new Float32Array([0.8, 0.9]);
            arraysEqual(output, expected);
        });
    });

    describe("removeEmotionMulti", () => {
        it("subtracts emotional value from embedding", () => {
            const input = [
                new Float32Array([1.0, 1.0]),
                new Float32Array([0.1, 0.1]),
            ];
            const output = model2d.removeEmotionMulti(input);
            const expected = [
                new Float32Array([0.8, 0.9]),
                new Float32Array([0.08, 0.09]),
            ]
            arraysEqual2d(output, expected);
        });
    });
});

const arraysEqual = (a: Float32Array, b: Float32Array) => {
    expect(a.length).toBe(b.length);
    const epsilon = 0.00001

    for (let i = 0; i < a.length; i++) {
        if (Math.abs(a[i] - b[i]) > epsilon) {
            fail(`At index ${i}: Expected ${a[i]} == ${b[i]}`);
        }
    }
}

const arraysEqual2d = (a: Float32Array[], b: Float32Array[]) => {
    expect(a.length).toBe(b.length);
    
    for (let i = 0; i < a.length; i++) {
        arraysEqual(a[i], b[i]);
    }
}