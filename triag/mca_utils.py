import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Union, Tuple, List, Callable, Dict

from torchvision.utils import save_image
from einops import rearrange, repeat


class AttentionBase:
    def __init__(self):
        self.cur_step = 0           # current denoising step
        self.num_att_layers = -1
        self.cur_att_layer = 0      # current attention layer index
        self._aux = {}

    def after_step(self):
        pass

    def __call__(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        out = self.forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            # after step
            self.after_step()
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=num_heads)
        return out

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0


class AttentionStore(AttentionBase):
    """
    Extension: Used to accumulate/store self-attention and cross-attention within a specified step range.
    - res: List of target resolutions (can be used to filter and retain attention at specific resolutions)
    - min_step/max_step: Only count attention within the range [min_step, max_step)

    Changed on 2025-08-29 to only store attention for conditional text
    """
    def __init__(self, res=[32], min_step=0, max_step=1000):
        super().__init__()
        self.res = res
        self.min_step = min_step
        self.max_step = max_step
        self.valid_steps = 0

        self.self_attns = []  # store the all attns
        self.cross_attns = []

        self.self_attns_step = []  # store the attns in each step
        self.cross_attns_step = []

        # To visualize cross/self-attention
        self.cross_attention_store = []
        self.self_attention_store = []

        # ★ Save only one snapshot (down/mid/up) for the Q visualization at "target_step".
        self.target_step = 30
        self.q_store = {"down": [], "mid": [], "up": []}

        # visualization for Value
        self.v_store = {"down": [], "mid": [], "up": []}
        
        # Visualize special steps for cross-attention
        self.cross_show_time = [5,10,15,20,25,30,35,40,45,50]
        self.cross_attention_store_list = []

    def get_average_attention(self):
        average_attention_self = self.self_attention_store
        average_attention_cross = self.cross_attention_store
        return [average_attention_self, average_attention_cross]
    
    def get_average_attention_list(self):
        average_attention_cross_list = self.cross_attention_store_list
        return average_attention_cross_list

    @staticmethod
    def get_empty_store():
        return []

    def after_step(self):
        if self.cur_step == self.target_step+1:
            self.self_attention_store = self.self_attns_step  # avoid OOM
        self.cross_attention_store = self.cross_attns_step
        if self.cur_step in self.cross_show_time:
            snap = [t.detach().cpu() for t in self.cross_attention_store]
            self.cross_attention_store_list.append(snap)
            # self.cross_attention_store_list.append(self.self_attention_store.detach().cpu())

        if self.cur_step > self.min_step and self.cur_step < self.max_step:
            self.valid_steps += 1
            # if len(self.self_attns) == 0:
            #     self.self_attns = self.self_attns_step
            #     self.cross_attns = self.cross_attns_step
            # else:
            #     for i in range(len(self.self_attns)):
            #         self.self_attns[i] += self.self_attns_step[i]
            #         self.cross_attns[i] += self.cross_attns_step[i]
        # self.self_attns_step.clear()  # 这个clear直接给所有的都弄没了
        # self.cross_attns_step.clear()
        self.self_attns_step = self.get_empty_store()
        self.cross_attns_step = self.get_empty_store()

    def storage_QKV(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads):
        if attn.shape[1] <= 64 ** 2:  # avoid OOM
            un_con = attn.shape[0]
            if is_cross and attn.shape[1] <= 32 ** 2:
                self.cross_attns_step.append(attn[un_con // 2:])
            if self.cur_step == self.target_step:
                self.self_attns_step.append(attn[un_con // 2:])

        if self.cur_step == self.target_step and not is_cross:
            # Only keep the resolutions you care about (optional)
            # If you only want to see specific res：if attn.shape[1] != res**2: pass
            qu, qc = q.chunk(2)
            B = qc.shape[0] // num_heads     # q: (B*H, N, D)
            N, D = qc.shape[1], qc.shape[2]
            q_bhnd = qc.detach().clone().view(B, num_heads, N, D)   # [B,H,N,D]
            q_mean = q_bhnd.mean(dim=1)                             # [B,N,D]
            self.q_store[place_in_unet].append(q_mean.detach().cpu())
            
            # Visualize Value 
            vu, vc = v.chunk(2)
            B = vc.shape[0] // num_heads     # q: (B*H, N, D)
            N, D = vc.shape[1], vc.shape[2]
            v_bhnd = vc.detach().clone().view(B, num_heads, N, D)   # [B,H,N,D]
            v_mean = v_bhnd.mean(dim=1)                             # [B,N,D]
            self.v_store[place_in_unet].append(v_mean.detach().cpu())


    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        if attn.shape[1] <= 64 ** 2:  # avoid OOM
            # Separate unconditional and conditional
            un_con = attn.shape[0]
            if is_cross:
                self.cross_attns_step.append(attn[un_con // 2:])
            else:
                self.self_attns_step.append(attn[un_con // 2:])
        if self.cur_step == self.target_step and not is_cross:
            qu, qc = q.chunk(2)
            B = qc.shape[0] // num_heads     # q: (B*H, N, D)
            N, D = qc.shape[1], qc.shape[2]
            q_bhnd = qc.detach().clone().view(B, num_heads, N, D)   # [B,H,N,D]
            q_mean = q_bhnd.mean(dim=1)                             # [B,N,D]
            self.q_store[place_in_unet].append(q_mean.detach().cpu())
            
            vu, vc = v.chunk(2)
            B = vc.shape[0] // num_heads     # q: (B*H, N, D)
            N, D = vc.shape[1], vc.shape[2]
            v_bhnd = vc.detach().clone().view(B, num_heads, N, D)   # [B,H,N,D]
            v_mean = v_bhnd.mean(dim=1)                             # [B,N,D]
            self.v_store[place_in_unet].append(v_mean.detach().cpu())

        return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
    

def regiter_attention_editor_diffusers(model, editor: AttentionBase):
    """
    Register a attention editor to Diffuser Pipeline, refer from [Prompt-to-Prompt]
    """
    def ca_forward(self, place_in_unet):
        def forward(x, encoder_hidden_states=None, attention_mask=None, context=None, mask=None):
            """
            The attention is similar to the original implementation of LDM CrossAttention class
            except adding some modifications on the attention
            """
            if encoder_hidden_states is not None:
                context = encoder_hidden_states
            if attention_mask is not None:
                mask = attention_mask

            to_out = self.to_out
            if isinstance(to_out, nn.modules.container.ModuleList):
                to_out = self.to_out[0]
            else:
                to_out = self.to_out

            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

            sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

            if mask is not None:
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            attn = sim.softmax(dim=-1)
            # the only difference
            out = editor(
                q, k, v, sim, attn, is_cross, place_in_unet,
                self.heads, scale=self.scale)

            return to_out(out)

        return forward

    def register_editor(net, count, place_in_unet):
        for name, subnet in net.named_children():
            if net.__class__.__name__ == 'Attention':  # spatial Transformer layer
                net.forward = ca_forward(net, place_in_unet)
                return count + 1
            elif hasattr(net, 'children'):
                count = register_editor(subnet, count, place_in_unet)
        return count

    cross_att_count = 0
    for net_name, net in model.unet.named_children():
        if "down" in net_name:
            cross_att_count += register_editor(net, 0, "down")
        elif "mid" in net_name:
            cross_att_count += register_editor(net, 0, "mid")
        elif "up" in net_name:
            cross_att_count += register_editor(net, 0, "up")
    editor.num_att_layers = cross_att_count


def regiter_attention_editor_ldm(model, editor: AttentionBase):
    """
    Register a attention editor to Stable Diffusion model, refer from [Prompt-to-Prompt]
    """
    def ca_forward(self, place_in_unet):
        def forward(x, encoder_hidden_states=None, attention_mask=None, context=None, mask=None):
            """
            The attention is similar to the original implementation of LDM CrossAttention class
            except adding some modifications on the attention
            """
            if encoder_hidden_states is not None:
                context = encoder_hidden_states
            if attention_mask is not None:
                mask = attention_mask

            to_out = self.to_out
            if isinstance(to_out, nn.modules.container.ModuleList):
                to_out = self.to_out[0]
            else:
                to_out = self.to_out

            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

            sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

            if mask is not None:
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            attn = sim.softmax(dim=-1)
            # the only difference
            out = editor(
                q, k, v, sim, attn, is_cross, place_in_unet,
                self.heads, scale=self.scale)

            return to_out(out)

        return forward

    def register_editor(net, count, place_in_unet):
        for name, subnet in net.named_children():
            if net.__class__.__name__ == 'CrossAttention':  # spatial Transformer layer
                net.forward = ca_forward(net, place_in_unet)
                return count + 1
            elif hasattr(net, 'children'):
                count = register_editor(subnet, count, place_in_unet)
        return count

    cross_att_count = 0
    for net_name, net in model.model.diffusion_model.named_children():
        if "input" in net_name:
            cross_att_count += register_editor(net, 0, "input")
        elif "middle" in net_name:
            cross_att_count += register_editor(net, 0, "middle")
        elif "output" in net_name:
            cross_att_count += register_editor(net, 0, "output")
    editor.num_att_layers = cross_att_count
