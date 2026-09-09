# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend-selection checks for NVFP4 KV cache on SM12x (consumer/workstation
Blackwell) via the FlashInfer FA2 paged kernels.

These run without a GPU: the device capability is monkeypatched on the
platform, and only class-level selection logic is exercised.
"""

from types import SimpleNamespace

import pytest

from vllm.config import set_current_vllm_config
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends import flashinfer as fi
from vllm.v1.attention.backends.flashinfer import FlashInferBackend
from vllm.v1.kv_cache_layout import KVCacheLayout


def _patch_capability(monkeypatch: pytest.MonkeyPatch, major: int, minor: int):
    cap = DeviceCapability(major, minor)
    monkeypatch.setattr(
        type(current_platform),
        "get_device_capability",
        classmethod(lambda cls, device_id=0: cap),
    )


def _fake_vllm_config(cache_dtype: str) -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype=cache_dtype),
        model_config=None,
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        attention_config=SimpleNamespace(use_trtllm_attention=None),
    )


@pytest.mark.parametrize("kv_cache_dtype", ["nvfp4", "nvfp4_4over6"])
def test_sm12x_supports_nvfp4_kv_without_trtllm(monkeypatch, kv_cache_dtype):
    """SM12x has no trtllm-gen; NVFP4 KV must still be accepted (FA2 reads it)."""
    _patch_capability(monkeypatch, 12, 0)
    monkeypatch.setattr(fi, "supports_trtllm_attention", lambda is_prefill: False)
    assert FlashInferBackend.supports_kv_cache_dtype(kv_cache_dtype)


@pytest.mark.parametrize("major,minor", [(12, 1)])
def test_sm121_supports_nvfp4_kv(monkeypatch, major, minor):
    _patch_capability(monkeypatch, major, minor)
    monkeypatch.setattr(fi, "supports_trtllm_attention", lambda is_prefill: False)
    assert FlashInferBackend.supports_kv_cache_dtype("nvfp4")


def test_sm100_nvfp4_kv_still_requires_trtllm(monkeypatch):
    """SM100 behaviour is unchanged: NVFP4 KV needs the trtllm-gen path."""
    _patch_capability(monkeypatch, 10, 0)
    monkeypatch.setattr(fi, "supports_trtllm_attention", lambda is_prefill: False)
    assert not FlashInferBackend.supports_kv_cache_dtype("nvfp4")
    monkeypatch.setattr(fi, "supports_trtllm_attention", lambda is_prefill: True)
    assert FlashInferBackend.supports_kv_cache_dtype("nvfp4")


@pytest.mark.parametrize("major,minor", [(9, 0), (8, 9), (8, 0)])
def test_pre_blackwell_rejects_nvfp4_kv(monkeypatch, major, minor):
    _patch_capability(monkeypatch, major, minor)
    monkeypatch.setattr(fi, "supports_trtllm_attention", lambda is_prefill: True)
    assert not FlashInferBackend.supports_kv_cache_dtype("nvfp4")


def test_sm12x_nvfp4_kv_prefers_head_major_layout(monkeypatch):
    """The FA2 NVFP4 paged reader takes the head-major block interior, same as
    trtllm-gen on SM100."""
    _patch_capability(monkeypatch, 12, 0)
    with set_current_vllm_config(_fake_vllm_config("nvfp4")):
        layouts = FlashInferBackend.supported_kv_cache_layouts()
    assert layouts == (KVCacheLayout.LBHNC, KVCacheLayout.BLHNC)


@pytest.mark.parametrize("cache_dtype", ["auto", "fp8", "fp8_e4m3"])
def test_sm12x_non_nvfp4_layout_unchanged(monkeypatch, cache_dtype):
    """fp8/auto on SM12x keep the default (no preference) layout resolution."""
    _patch_capability(monkeypatch, 12, 0)
    with set_current_vllm_config(_fake_vllm_config(cache_dtype)):
        layouts = FlashInferBackend.supported_kv_cache_layouts()
    assert layouts is None


def test_sm100_layout_unchanged(monkeypatch):
    _patch_capability(monkeypatch, 10, 0)
    with set_current_vllm_config(_fake_vllm_config("nvfp4")):
        layouts = FlashInferBackend.supported_kv_cache_layouts()
    assert layouts == (KVCacheLayout.LBHNC, KVCacheLayout.BLHNC)
