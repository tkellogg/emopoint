import dataclasses

import numpy as np


@dataclasses.dataclass
class DimLabel:
    negative: str
    positive: str

    @property
    def label(self) -> str:
        return f"{self.negative}<->{self.positive}"

    def __str__(self) -> str:
        return self.label


@dataclasses.dataclass
class EmoModel:
    weights: np.ndarray
    dims: list[DimLabel]
    num_emb_dims: int

    def emb_to_emo(self, emb: np.ndarray | list[float]) -> np.ndarray:
        """
        Map an embedding downward into emotion space. An embedding is a vector of
        length num_emb_dims, typically hudreds or thousands of values. The output
        is a much smaller vector representing only the emotional aspects of the text.
        """
        if isinstance(emb, list):
            emb = np.array(emb)

        if not isinstance(emb, np.ndarray):
            raise ValueError("Input must be a list or numpy array.")

        if len(emb.shape) > 2:
            raise ValueError("Input must be 1- or 2-dimensional.")
        elif len(emb.shape) == 2:
            if emb.shape[0] == self.num_emb_dims:
                return self.weights @ emb
            elif emb.shape[1] == self.num_emb_dims:
                return self.weights @ emb.T
            else:
                raise ValueError(f"Input must be an array of shape (*, {self.num_emb_dims}) or ({self.num_emb_dims}, *).")
        else:
            if emb.shape[0] == self.num_emb_dims:
                return self.weights @ emb
            else:
                raise ValueError(f"Input must be an array of shape (*, {self.num_emb_dims}).")
