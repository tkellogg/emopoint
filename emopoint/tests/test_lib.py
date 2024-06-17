import numpy as np
from emopoint.lib import EmoModel, DimLabel


def test_json_io():
    em = EmoModel(
        weights=np.array([
            [1.0, 2.0],
            [1.2, 2.2],
            [1.3, 2.3],
        ]),
        dims=[
            DimLabel("bad", "good"),
            DimLabel("kinda", "totally"),
            DimLabel("fad", "historical"),
        ],
        num_emb_dims=2,
    )
    recreated = EmoModel.from_json(em.to_json())
    assert em == recreated


def test_remove_emo_2d():
    em = EmoModel(
        weights=np.array([
            [0.1, 0.1],
            [0.0, 0.0],
            [0.1, 0.0],
        ]),
        dims=[DimLabel("", "") for _ in range(3)],
        num_emb_dims=2,
    )
    orig = np.array([1.0, 1.0])
    res = em.remove_emo(orig)
    assert orig.shape == res.shape
    assert np.all(res == np.array([0.8, 0.9]))


def test_remove_emo_2d_2_examples():
    em = EmoModel(
        weights=np.array([
            [0.1, 0.1],
            [0.0, 0.0],
            [0.1, 0.0],
        ]),
        dims=[DimLabel("", "") for _ in range(3)],
        num_emb_dims=2,
    )
    orig = [np.array([1.0, 1.0]), np.array([1.0, 1.0])]
    res = em.remove_emo(orig)
    assert (2, 2) == res.shape
    assert np.all(res == np.array([[0.8, 0.9], [0.8, 0.9]]))


def test_remove_emo_4d():
    em = EmoModel(
        weights=np.array([
            [0.1, 0.1, 0.0, 0.0],
            [0.0, 0.0, 0.1, 0.0],
            [0.1, 0.0, 0.0, 0.1],
        ]),
        dims=[DimLabel("", "") for _ in range(3)],
        num_emb_dims=4,
    )
    orig = np.array([1.0, 1.0, 1.0, 1.0])
    res = em.remove_emo(orig)
    assert orig.shape == res.shape
    assert np.all(res == np.array([0.8, 0.9, 0.9, 0.9]))