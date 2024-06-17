export class DimLabel {
    readonly negative: string;
    readonly positive: string;

    constructor(negative: string, positive: string) {
        this.negative = negative;
        this.positive = positive;
    }

    getLabel(): string {
        return `${this.negative}<->${this.positive}`
    }
}


export class EmoModel {
    readonly weights: Float32Array[];
    readonly dims: DimLabel[];
    readonly numberEmbeddingDimensions: number;

    constructor(weights: Float32Array[], dims: DimLabel[], numberEmbeddingDimensions: number) {
        this.weights = weights;
        this.dims = dims;
        this.numberEmbeddingDimensions = numberEmbeddingDimensions;
    }

    embeddingToEmopoint(embedding: Float32Array): Float32Array {
        if (embedding.length != this.numberEmbeddingDimensions) {
            throw new Error(`Expected embedding of length ${this.numberEmbeddingDimensions}`)
        }

        // Matrix * Vector multiply
        const res = new Float32Array(this.weights.length);
        for (let iDim=0; iDim<this.weights.length; iDim++) {
            for (let iEmb=0; iEmb<embedding.length; iEmb++) {
                res[iDim] += embedding[iEmb] * this.weights[iDim][iEmb];
            }
        }

        return res;
    }

    embeddingToEmopointMulti(embedding: Float32Array[]): Float32Array[] {
        return embedding.map(e => this.embeddingToEmopoint(e));
    }

    removeEmotion(embedding: Float32Array): Float32Array {
        if (embedding.length != this.numberEmbeddingDimensions) {
            throw new Error(`Expected embedding of length ${this.numberEmbeddingDimensions}`)
        }

        const res = new Float32Array(this.numberEmbeddingDimensions);
        for (let iEmb=0; iEmb<embedding.length; iEmb++) {
            for (let iDim=0; iDim<this.weights.length; iDim++) {
                res[iEmb] += embedding[iEmb] * this.weights[iDim][iEmb];
            }
            res[iEmb] = embedding[iEmb] - res[iEmb];
        }

        return res;
    }

    removeEmotionMulti(embedding: Float32Array[]): Float32Array[] {
        return embedding.map(e => this.removeEmotion(e));
    }
}