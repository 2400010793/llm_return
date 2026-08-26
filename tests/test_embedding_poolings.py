import torch

from src.text.embeddings import pool_hidden_states


def test_pool_hidden_states_excludes_padding_and_special_tokens():
    hidden = torch.tensor([[[1.0], [2.0], [6.0], [100.0], [200.0]]])
    attention = torch.tensor([[1, 1, 1, 1, 0]])
    special = torch.tensor([[1, 0, 0, 1, 1]])

    pooled = pool_hidden_states(hidden, attention, special)

    assert pooled["mean"].tolist() == [[4.0]]
    assert pooled["cls"].tolist() == [[1.0]]
    assert pooled["max"].tolist() == [[6.0]]


def test_pool_hidden_states_zeroes_empty_content():
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    attention = torch.tensor([[1, 1]])
    special = torch.tensor([[1, 1]])

    pooled = pool_hidden_states(hidden, attention, special)

    assert pooled["mean"].tolist() == [[0.0, 0.0]]
    assert pooled["max"].tolist() == [[0.0, 0.0]]
    assert pooled["cls"].tolist() == [[1.0, 2.0]]
