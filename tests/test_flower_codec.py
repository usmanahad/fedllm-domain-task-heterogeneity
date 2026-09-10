from __future__ import annotations

import torch

from fedllm_heterogeneity.flower_app import codec_from_adapter


def test_codec_round_trip():
    adapter = {
        "layer.lora_A.default.weight": torch.randn(2, 5),
        "layer.lora_B.default.weight": torch.randn(4, 2),
    }
    codec = codec_from_adapter(adapter, scaling=2.0)
    residual = {"layer": torch.randn(4, 5)}
    decoded_adapter, decoded_residual = codec.decode(codec.encode(adapter, residual))
    assert torch.equal(decoded_adapter["layer.lora_A.default.weight"], adapter["layer.lora_A.default.weight"])
    assert torch.equal(decoded_residual["layer"], residual["layer"])

