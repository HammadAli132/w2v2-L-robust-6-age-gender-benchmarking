"""Original heads plus attention masking and masked temporal pooling."""
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import load_file
from transformers import Wav2Vec2Config, Wav2Vec2Model

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
CHECKPOINT = ARTIFACTS / "checkpoint"
SAMPLE_RATE = 16000
LABELS = ["female", "male", "child"]


class Head(nn.Module):
    def __init__(self, size, outputs):
        super().__init__()
        self.dense = nn.Linear(size, size)
        self.out_proj = nn.Linear(size, outputs)

    def forward(self, x):
        return self.out_proj(torch.tanh(self.dense(x)))


class AgeGender(nn.Module):
    def __init__(self, attention="sdpa"):
        super().__init__()
        config = Wav2Vec2Config.from_pretrained(CHECKPOINT, local_files_only=True)
        config._attn_implementation = attention
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = Head(config.hidden_size, 1)
        self.gender = Head(config.hidden_size, 3)

    def forward(self, signal, lengths):
        mask = torch.arange(signal.shape[1], device=signal.device)[None, :] < lengths[:, None]
        hidden = self.wav2vec2(signal, attention_mask=mask.long()).last_hidden_state
        feature_lengths = self.wav2vec2._get_feat_extract_output_lengths(lengths)
        valid = torch.arange(hidden.shape[1], device=hidden.device)[None, :] < feature_lengths[:, None]
        pooled = (hidden * valid.unsqueeze(-1)).sum(1) / feature_lengths[:, None].to(hidden.dtype)
        return self.age(pooled), self.gender(pooled)


def load_model(device="cpu", attention="sdpa"):
    model = AgeGender(attention)
    state = load_file(str(CHECKPOINT / "model.safetensors"))
    # Original Transformers used legacy weight_norm names.
    for old, new in (("weight_g", "parametrizations.weight.original0"),
                     ("weight_v", "parametrizations.weight.original1")):
        key = f"wav2vec2.encoder.pos_conv_embed.conv.{old}"
        if key in state:
            state[f"wav2vec2.encoder.pos_conv_embed.conv.{new}"] = state.pop(key)
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)
