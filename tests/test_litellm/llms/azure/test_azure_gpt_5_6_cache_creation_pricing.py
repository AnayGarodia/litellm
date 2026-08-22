import pytest

import litellm
from litellm import get_model_info
from litellm.litellm_core_utils.get_model_cost_map import get_model_cost_map


@pytest.fixture(autouse=True)
def reload_model_costs():
    litellm.model_cost = get_model_cost_map(url=None)
    yield


@pytest.mark.parametrize(
    "model,expected_cache_creation_cost",
    [
        ("azure/gpt-5.6", 6.25e-06),
        ("azure/gpt-5.6-sol", 6.25e-06),
        ("azure/gpt-5.6-terra", 2.5e-06),
        ("azure/gpt-5.6-luna", 2.5e-07),
        ("azure/us/gpt-5.6", 6.875e-06),
        ("azure/us/gpt-5.6-sol", 6.875e-06),
        ("azure/us/gpt-5.6-terra", 2.75e-06),
        ("azure/us/gpt-5.6-luna", 2.75e-07),
        ("azure/eu/gpt-5.6", 6.875e-06),
        ("azure/eu/gpt-5.6-sol", 6.875e-06),
        ("azure/eu/gpt-5.6-terra", 2.75e-06),
        ("azure/eu/gpt-5.6-luna", 2.75e-07),
    ],
)
def test_azure_gpt_5_6_cache_creation_pricing(model, expected_cache_creation_cost):
    model_info = get_model_info(model=model)

    assert model_info.get("cache_creation_input_token_cost") is not None
    assert model_info["cache_creation_input_token_cost"] == expected_cache_creation_cost


@pytest.mark.parametrize(
    "model,cache_creation_tokens,expected_cost",
    [
        ("azure/gpt-5.6", 1000, 1000 * 6.25e-06),
        ("azure/us/gpt-5.6-luna", 1000, 1000 * 2.75e-07),
    ],
)
def test_azure_gpt_5_6_cache_creation_completion_is_priced(model, cache_creation_tokens, expected_cost):
    from litellm.cost_calculator import completion_cost
    from litellm.types.utils import ModelResponse, Usage

    response = ModelResponse(
        model=model,
        choices=[{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop", "index": 0}],
        usage=Usage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=cache_creation_tokens,
            cache_creation_input_tokens=cache_creation_tokens,
        ),
    )

    cost = completion_cost(completion_response=response, model=model)

    assert cost == pytest.approx(expected_cost)
