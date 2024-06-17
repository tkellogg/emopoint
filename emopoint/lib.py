import dataclasses
import functools

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

    def to_json(self) -> str:
        import json
        d = dataclasses.asdict(self)
        d["weights"] = d["weights"].tolist()
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "EmoModel":
        import json
        d = json.loads(json_str)
        d["weights"] = np.array(d["weights"])
        return cls(
            weights=d["weights"],
            dims=[DimLabel(**dim) for dim in d["dims"]],
            num_emb_dims=d["num_emb_dims"],
        )

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, EmoModel):
            return False
        return (
            self.num_emb_dims == value.num_emb_dims and
            all(map(lambda a: a[0] == a[1], zip(self.dims, value.dims))) and
            np.all(self.weights == value.weights)
        )

    @property
    def weights_1d(self) -> np.ndarray:
        if not hasattr(self, "_weights_1d"):
            if self.num_emb_dims == 1:
                val = self.weights
            else:
                val = functools.reduce(
                    lambda acc, dim: acc + dim,
                    self.weights,
                )
            setattr(self, "_weights_1d", val)
        return getattr(self, "_weights_1d")

    def emb_to_emo(self, emb: np.ndarray | list[float]) -> np.ndarray:
        """
        Map an embedding downward into emotion space. An embedding is a vector of
        length num_emb_dims, typically hudreds or thousands of values. The output
        is a much smaller vector representing only the emotional aspects of the text.
        """
        arg = self._check_args(emb)
        return (self.weights @ arg).T

    def _check_args(self, emb: np.ndarray | list[float]) -> np.ndarray:
        if isinstance(emb, list):
            emb = np.array(emb)

        if not isinstance(emb, np.ndarray):
            raise ValueError("Input must be a list or numpy array.")

        if len(emb.shape) > 2:
            raise ValueError("Input must be 1- or 2-dimensional.")
        elif len(emb.shape) == 1:
            emb = emb.reshape(1, -1)

        if len(emb.shape) == 2:
            if emb.shape[0] == self.num_emb_dims:
                return emb
            elif emb.shape[1] == self.num_emb_dims:
                return emb.T
            else:
                raise ValueError(f"Input must be an array of shape (*, {self.num_emb_dims}) or ({self.num_emb_dims}, *).")
        else:
            if emb.shape[0] == self.num_emb_dims:
                return emb
            else:
                raise ValueError(f"Input must be an array of shape (*, {self.num_emb_dims}).")

    def remove_emo(self, emb: np.ndarray | list[float]) -> np.ndarray:
        """
        Take an embedding or a set of embeddings and remove the emotional
        information, returning a new set of embeddings in the same dimensionality
        but with emotion removed.
        """
        arg = self._check_args(emb)
        return np.array([
            arg[:, i] - (self.weights_1d * arg[:, i])
            for i in range(arg.shape[1])
        ])

    def magnitude(self, emb: np.ndarray | list[float]) -> list[float]:
        """
        
        """
        emo = self.emb_to_emo(emb)
        return [np.linalg.norm(x) for x in emo]
        
