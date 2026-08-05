"""
Tests for `litellm/router_strategy/lowest_cost.py`.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.caching.caching import DualCache
from litellm.router_strategy.lowest_cost import LowestCostLoggingHandler


@pytest.mark.asyncio
async def test_provider_prefixed_model_resolves_real_cost():
    test_cache = DualCache()
    model_list = [
        {
            "model_name": "chat-group",
            "litellm_params": {"model": "openai/gpt-4"},
            "model_info": {"id": "expensive-deployment"},
        },
        {
            "model_name": "chat-group",
            "litellm_params": {"model": "openai/gpt-4o-mini"},
            "model_info": {"id": "cheap-deployment"},
        },
    ]
    lowest_cost_logger = LowestCostLoggingHandler(router_cache=test_cache)

    selected_deployment = await lowest_cost_logger.async_get_available_deployments(
        model_group="chat-group", healthy_deployments=model_list
    )

    assert selected_deployment["model_info"]["id"] == "cheap-deployment"
