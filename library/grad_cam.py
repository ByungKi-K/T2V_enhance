import re
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional
from diffusers.models.attention_processor import AttnProcessor2_0

class SaveGradCrossAttnProcessor(AttnProcessor2_0):
    """cross-attn에서 softmax(QK^T) 확률 probs를 만들고 requires_grad_(True)로 저장."""
    def __init__(self, store, name):
        super().__init__()
        self.store = store
        self.name = name

    def forward(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, scale=1.0):
        if encoder_hidden_states is None:
            return super().forward(attn, hidden_states, encoder_hidden_states, attention_mask, temb, scale)
        residual = hidden_states if attn.residual_connection else None

        Q = attn.head_to_batch_dim(attn.to_q(hidden_states))
        K = attn.head_to_batch_dim(attn.to_k(encoder_hidden_states))
        V = attn.head_to_batch_dim(attn.to_v(encoder_hidden_states))

        scale_fac = getattr(attn, "scale", Q.shape[-1] ** -0.5)
        scores = Q @ K.transpose(-1, -2) * scale_fac
        if attention_mask is not None:
            scores = scores + attention_mask

        probs = scores.softmax(dim=-1)  # (B*H, Q, Ktxt)
        probs.requires_grad_(True)

        out = probs @ V
        out = attn.batch_to_head_dim(out)
        out = attn.to_out[0](out)
        if attn.use_out_norm:
            out = attn.to_out[1](out)
        if residual is not None:
            out = out + residual

        # GPU에 둬야 grad가 살아있음
        self.store.append({"name": self.name, "probs": probs})
        return out

def _find_token_indices(tokens: List[str], keywords: List[str]):
    kws = [k.lower() for k in keywords]
    idxs = []
    for i, t in enumerate(tokens):
        tn = t.lower().replace("▁","").replace("</w>","")
        if any(k in tn for k in kws):
            idxs.append(i)
    return sorted(set(idxs))

def _reshape_q_to_thw(q_vec: torch.Tensor, Tprime: int, Hprime: int, Wprime: int):
    x = q_vec.reshape(Tprime, Hprime, Wprime)
    x = (x - x.min()) / (x.max() - x.min() + 1e-6)
    return x

def _upsample_3d_to_video(x_thw: torch.Tensor, T_full: int, H_full: int, W_full: int, spa_down: int):
    x = x_thw[None, None, ...]  # (1,1,T',H',W')
    x = F.interpolate(x, size=(T_full, H_full//spa_down, W_full//spa_down), mode="trilinear", align_corners=False)
    x = F.interpolate(x.squeeze(0), size=(H_full, W_full), mode="bilinear", align_corners=False)  # (1,T,H,W)->(T,H,W)
    return x.squeeze(0)  # (T,H,W)