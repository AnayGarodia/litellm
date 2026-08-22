import pytest

import litellm
from litellm import get_model_info
from litellm.litellm_core_utils.get_model_cost_map import get_model_cost_map


@pytest.fixture(autouse=True)
def reload_model_costs():
    litellm.model_cost = get_model_cost_map(url=None)
    yield


@pytest.mark.parametrize("region", ["us", "eu"])
def test_azure_gpt4o_2024_11_20_regional_cache_read_pricing(region):
    model_info = get_model_info(model=f"azure/{region}/gpt-4o-2024-11-20")

    assert model_info.get("cache_read_input_token_cost") is not None
    assert model_info["cache_read_input_token_cost"] == 1.375e-06
    assert model_info.get("supports_prompt_caching") is True


@pytest.mark.parametrize("region", ["us", "eu"])
def test_azure_gpt4o_2024_11_20_cached_completion_is_priced(region):
    from litellm.cost_calculator import completion_cost
    from litellm.types.utils import ModelResponse, Usage

    response = ModelResponse(
        model=f"azure/{region}/gpt-4o-2024-11-20",
        choices=[{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop", "index": 0}],
        usage=Usage(
            prompt_tokens=1000,
            completion_tokens=10,
            total_tokens=1010,
            prompt_tokens_details={"cached_tokens": 1000},
        ),
    )

    cost = completion_cost(completion_response=response, model=f"azure/{region}/gpt-4o-2024-11-20")

    assert cost > 0
    assert cost == pytest.approx(1000 * 1.375e-06 + 10 * 1.1e-05)
