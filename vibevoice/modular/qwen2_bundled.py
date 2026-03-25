"""
Qwen2 compatibility shim for VibeVoice.

Imports Qwen2 classes from the installed transformers package.
Longform models require transformers>=4.51.3,<5.0.0.
"""

from transformers.models.qwen2.modeling_qwen2 import Qwen2Model, Qwen2PreTrainedModel
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

try:
    from transformers.models.llama.modeling_llama import LlamaRMSNorm as Qwen2RMSNorm
except ImportError:
    import torch
    import torch.nn as nn

    class Qwen2RMSNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.variance_epsilon = eps

        def forward(self, hidden_states):
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            return self.weight * hidden_states.to(input_dtype)


def get_qwen2_model_class():
    return Qwen2Model
