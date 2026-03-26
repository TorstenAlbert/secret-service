"""Tests for EmbeddingEncoder."""
from __future__ import annotations

import struct

import numpy as np
import pytest

from ss.vectors.encoder import EmbeddingEncoder


EXPECTED_DIM = 384


@pytest.fixture
def encoder() -> EmbeddingEncoder:
    return EmbeddingEncoder()


# ---------------------------------------------------------------------------
# Lazy loading
# ---------------------------------------------------------------------------

def test_model_starts_unloaded():
    enc = EmbeddingEncoder()
    assert enc._model is None


def test_model_loads_on_first_encode(encoder: EmbeddingEncoder):
    encoder.encode("hello world")
    assert encoder._model is not None


def test_model_loaded_only_once(encoder: EmbeddingEncoder):
    encoder.encode("first call")
    model_ref = encoder._model
    encoder.encode("second call")
    assert encoder._model is model_ref


# ---------------------------------------------------------------------------
# Dimension property
# ---------------------------------------------------------------------------

def test_dimension_property(encoder: EmbeddingEncoder):
    assert encoder.dimension == EXPECTED_DIM


# ---------------------------------------------------------------------------
# encode() shape and type
# ---------------------------------------------------------------------------

def test_encode_returns_ndarray(encoder: EmbeddingEncoder):
    result = encoder.encode("test sentence")
    assert isinstance(result, np.ndarray)


def test_encode_correct_shape(encoder: EmbeddingEncoder):
    result = encoder.encode("test sentence")
    assert result.shape == (EXPECTED_DIM,)


def test_encode_is_normalized(encoder: EmbeddingEncoder):
    result = encoder.encode("normalized check")
    norm = float(np.linalg.norm(result))
    assert abs(norm - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# encode_bytes()
# ---------------------------------------------------------------------------

def test_encode_bytes_returns_bytes(encoder: EmbeddingEncoder):
    result = encoder.encode_bytes("test")
    assert isinstance(result, bytes)


def test_encode_bytes_correct_length(encoder: EmbeddingEncoder):
    result = encoder.encode_bytes("test sentence")
    # float32 = 4 bytes each
    assert len(result) == EXPECTED_DIM * 4


def test_encode_bytes_is_float32(encoder: EmbeddingEncoder):
    result = encoder.encode_bytes("test")
    floats = struct.unpack(f"{EXPECTED_DIM}f", result)
    assert len(floats) == EXPECTED_DIM


def test_encode_bytes_matches_encode(encoder: EmbeddingEncoder):
    text = "consistency check"
    arr = encoder.encode(text)
    b = encoder.encode_bytes(text)
    floats = struct.unpack(f"{EXPECTED_DIM}f", b)
    arr_from_bytes = np.array(floats, dtype=np.float32)
    np.testing.assert_allclose(arr.astype(np.float32), arr_from_bytes, rtol=1e-5)


# ---------------------------------------------------------------------------
# Semantic similarity
# ---------------------------------------------------------------------------

def test_similar_texts_high_cosine_similarity(encoder: EmbeddingEncoder):
    a = encoder.encode("The cat sat on the mat")
    b = encoder.encode("A cat is sitting on a mat")
    cos_sim = float(np.dot(a, b))  # already normalized
    assert cos_sim > 0.7, f"Expected high similarity, got {cos_sim:.4f}"


def test_different_texts_lower_similarity(encoder: EmbeddingEncoder):
    a = encoder.encode("The cat sat on the mat")
    b = encoder.encode("quantum physics research paper")
    cos_sim = float(np.dot(a, b))
    # Should be notably lower than similar texts
    assert cos_sim < 0.6, f"Expected lower similarity, got {cos_sim:.4f}"


def test_similar_texts_more_similar_than_different(encoder: EmbeddingEncoder):
    anchor = encoder.encode("machine learning model training")
    similar = encoder.encode("training a neural network model")
    different = encoder.encode("medieval history of france")
    sim_close = float(np.dot(anchor, similar))
    sim_far = float(np.dot(anchor, different))
    assert sim_close > sim_far
