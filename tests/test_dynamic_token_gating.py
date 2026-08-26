import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.models.dynamic_token_gating import TokenVariableSelectionNetwork


def test_dynamic_weights_sum_to_one_and_masked_tokens_are_zero() -> None:
    torch.manual_seed(42)
    model = TokenVariableSelectionNetwork(4, 3, gate_hidden_size=8)
    tokens = torch.randn(2, 3, 4)
    mask = torch.tensor([[True, True, False], [True, False, True]])
    output = model(tokens, mask)
    torch.testing.assert_close(output.weights.sum(dim=1), torch.ones(2))
    assert output.weights[0, 2].item() == 0.0
    assert output.weights[1, 1].item() == 0.0
    assert output.prediction.shape == (2,)
    assert output.representation.shape == (2, 64)


def test_dynamic_gate_is_announcement_specific() -> None:
    torch.manual_seed(7)
    model = TokenVariableSelectionNetwork(
        3, 3, gate_hidden_size=6, representation_size=5, gate_mode="dynamic"
    )
    first = torch.tensor([[[-4.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]])
    second = torch.tensor([[[4.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, -1.0]]])
    weights = model(torch.cat((first, second), dim=0)).weights.detach().numpy()
    assert not np.allclose(weights[0], weights[1])


def test_uniform_gate_is_exact_ablation_over_unmasked_tokens() -> None:
    model = TokenVariableSelectionNetwork(2, 4, gate_mode="uniform")
    mask = torch.tensor([[True, False, True, True]])
    weights = model(torch.randn(1, 4, 2), mask).weights
    torch.testing.assert_close(
        weights, torch.tensor([[1.0 / 3.0, 0.0, 1.0 / 3.0, 1.0 / 3.0]])
    )


def test_static_gate_does_not_depend_on_announcement_values() -> None:
    torch.manual_seed(3)
    model = TokenVariableSelectionNetwork(2, 3, gate_mode="static")
    weights = model(torch.randn(5, 3, 2)).weights
    torch.testing.assert_close(weights, weights[0].expand_as(weights))


def test_gate_rejects_fully_masked_announcement() -> None:
    model = TokenVariableSelectionNetwork(2, 3)
    with pytest.raises(ValueError, match="at least one token"):
        model(torch.randn(1, 3, 2), torch.zeros(1, 3, dtype=torch.bool))
