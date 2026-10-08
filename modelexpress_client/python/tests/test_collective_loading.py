# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from modelexpress.adapter import EngineAdapter, StrategyFailed, StrategyRecoveryError
from modelexpress.engines.sglang.adapter import SglangAdapter
from modelexpress.load_strategy import execute_load_strategies
from modelexpress.load_strategy.base import LoadStrategy


class CollectiveAdapter(EngineAdapter):
    collective_loading = True

    def all_gather_state(self, state):
        world = SimpleNamespace(cpu_group=dist.group.WORLD)
        module = SimpleNamespace(
            get_parallel=lambda: SimpleNamespace(world_group=world)
        )
        with patch.dict(sys.modules, {"sglang.srt.runtime_context": module}):
            return SglangAdapter.all_gather_state(self, state)

    def reinit_for_retry(self, result):
        result.value.fill_(0)
        return result


class Peer(LoadStrategy):
    name = "rdma"

    def __init__(self, scenario):
        self.scenario = scenario

    def is_available(self, ctx):
        return self.scenario != "unavailable" or ctx.global_rank != 0

    def load(self, result, ctx):
        if self.scenario == "cold" or (
            ctx.global_rank == 0 and self.scenario != "warm"
        ):
            if self.scenario == "fatal":
                raise StrategyRecoveryError("broken model")
            if self.scenario == "partial":
                result.value.fill_(-1)
                raise StrategyFailed("interrupted transfer", mutated=True)
            raise StrategyFailed("missing metadata")
        result.value.fill_(7)
        return result

    def rollback(self, ctx):
        pass


class Disk(LoadStrategy):
    name = "default"

    def is_available(self, ctx):
        return True

    def load(self, result, ctx):
        assert result.value.item() == 0
        dist.barrier()
        result.value.fill_(9)
        return result


def _run_rank(rank, rendezvous, scenario):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=3,
        timeout=timedelta(seconds=15),
    )
    try:
        ctx = SimpleNamespace(
            adapter=CollectiveAdapter(),
            global_rank=rank,
            identity=SimpleNamespace(model_name="test"),
        )
        model = torch.zeros(1)
        with patch("modelexpress.load_strategy.publish_source_if_supported") as publish:
            if scenario == "fatal":
                with pytest.raises(StrategyRecoveryError):
                    execute_load_strategies(model, ctx, [Peer(scenario), Disk()])
                publish.assert_not_called()
            else:
                loaded = execute_load_strategies(model, ctx, [Peer(scenario), Disk()])
                assert loaded.item() == (7 if scenario == "warm" else 9)
                publish.assert_called_once()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "scenario", ["cold", "warm", "metadata", "partial", "fatal", "unavailable"]
)
def test_collective_loading(tmp_path, scenario):
    mp.spawn(_run_rank, args=(f"file://{tmp_path / 'rendezvous'}", scenario), nprocs=3)
