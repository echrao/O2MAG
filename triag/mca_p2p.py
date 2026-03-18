import os

import torch
import torch.nn.functional as F
import numpy as np
import math

from einops import rearrange

from .mca_utils import AttentionBase, AttentionStore

from torchvision.utils import save_image

from .seq_aligner import get_replacement_mapper

from .ptp_utils import get_time_words_attention_alpha, get_word_inds

from .ptp_utils import report_row_sums

MAX_NUM_WORDS = 77

# If you want to visualize it, remember to replace AttentionBase with AttentionStore
class McaControlReplace(AttentionBase):
    MODEL_TYPE = {
        "SD": 16,
        "SDXL": 70
    }

    def __init__(self, prompts, tokenizer, replace_idx, self_replace_steps, cross_replace_steps, equalizer, start_step=4, end_step=50, start_layer=10, end_layer=None, layer_idx=None, step_idx=None, total_steps=50, mask_s=None, mask_t=None, mask_save_dir=None, model_type="SD", attn_store_judge=False, self_attn_reweight=1.1):
        """
        Maske-guided MasaCtrl to alleviate the problem of fore- and background confusion
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            mask_s: source mask with shape (h, w)
            mask_t: target mask with same shape as source mask
            mask_save_dir: the path to save the mask image
            model_type: the model type, SD or SDXL
        """
        super().__init__()
        self.mask_s = mask_s    # Mask of the reference anomalous image
        self.mask_t = mask_t    # Mask of the expected anomalous image of the target image
        self.total_steps = total_steps
        self.total_layers = self.MODEL_TYPE.get(model_type, 16)
        self.start_step = start_step
        self.start_layer = start_layer
        self.end_step = end_step
        if end_layer is None:
            self.end_layer = self.total_layers
        else:
            self.end_layer = end_layer
        self.replace_idx = replace_idx
        # self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, end_layer))
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, self.end_layer))
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, end_step))
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps  # self_replace_steps = 0,0.1
        if type(cross_replace_steps) is float:
            cross_replace_steps = 0, cross_replace_steps  # cross_replace_steps = 0,0.8
        self.num_self_replace = int(total_steps * self_replace_steps[0]), int(total_steps * self_replace_steps[1])
        self.num_cross_replace = int(total_steps * cross_replace_steps[0]), int(total_steps * cross_replace_steps[1])
        self.equalizer = equalizer
        # avoid OOM
        self.attn_store_judge = attn_store_judge
        self.self_attn_reweight = self_attn_reweight
        # print("TriAG: ",self.num_self_replace)
        print("DAE attention re-weight timestep: ",self.num_cross_replace)
        print("TriAG at denoising steps: ", self.step_idx)
        print("TriAG at U-Net layers: ", self.layer_idx)
        if mask_save_dir is not None:
            os.makedirs(mask_save_dir, exist_ok=True)
            save_image(self.mask_s.unsqueeze(0).unsqueeze(0), os.path.join(mask_save_dir, "mask_s.png"))
            save_image(self.mask_t.unsqueeze(0).unsqueeze(0), os.path.join(mask_save_dir, "mask_t.png"))
    
    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        B = q.shape[0] // num_heads
        H = W = int(np.sqrt(q.shape[1]))
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k_f = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v_f = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)
        if kwargs.get("is_mask_attn"):
            # print('Mask self-attention......')
            k_t = rearrange(kwargs.get("k_bg"), "(b h) n d -> h (b n) d", h=num_heads)
            v_t = rearrange(kwargs.get("v_bg"), "(b h) n d -> h (b n) d", h=num_heads)

            mask_t = self.mask_t.unsqueeze(0).unsqueeze(0)
            mask_t = F.interpolate(mask_t, (H, W)).flatten(0).unsqueeze(0)
            mask_t = mask_t.flatten()

            mask_r = self.mask_s.to(q.dtype).to(q.device)[None, None, ...]     # [1,1,512,512]
            mask_r = F.adaptive_max_pool2d(mask_r, (H, W))                          # [1,1,H,W]
            mask_r = mask_r.view(1, -1).squeeze(0) 

            sim_fg = torch.einsum("h i d, h j d -> h i j", q, k_f) * kwargs.get("scale")
            sim_fg = sim_fg + mask_r.masked_fill(mask_r == 0, torch.finfo(sim.dtype).min)

            # ---- self-attention enhancement
            sim_fg = sim_fg + mask_r.view(1,1,-1) * math.log(self.self_attn_reweight)
            sim_bg = torch.einsum("h i d, h j d -> h i j", q, k_t) * kwargs.get("scale")
            sim_bg = sim_bg + mask_t.masked_fill(mask_t == 1, torch.finfo(sim.dtype).min)

            attn_fg = sim_fg.softmax(-1)
            tau_fg = 0.7
            attn_fg = (sim_fg / tau_fg).softmax(-1)   # tau_fg ≈ 0.7
            attn_bg = sim_bg.softmax(-1)

            out_fg = torch.einsum("h i j, h j d -> h i d", attn_fg, v_f)
            out_bg = torch.einsum("h i j, h j d -> h i d", attn_bg, v_t)

            out = torch.cat([out_fg, out_bg])

            out = rearrange(out, "(h1 h) (b n) d -> (h1 b) n (h d)", b=B, h=num_heads)
            
            return out
        
        sim = torch.einsum("h i d, h j d -> h i j", q, k_f) * kwargs.get("scale")
        attn = sim.softmax(-1)
        if len(attn) == 2 * len(v_f):
            v_f = torch.cat([v_f] * 2)
        out = torch.einsum("h i j, h j d -> h i d", attn, v_f)
        out = rearrange(out, "(h1 h) (b n) d -> (h1 b) n (h d)", b=B, h=num_heads)
        return out

    def attn_batch2(self, q_bg, k_bg, q_fg, k_fg, v_bg, v_fg, num_heads, **kwargs):
        """
        Performing attention for a batch of queries, keys, and values
        """
        B = q_bg.shape[0] // num_heads
        H = W = int(np.sqrt(q_bg.shape[1]))
        q_bg = rearrange(q_bg, "(b h) n d -> h (b n) d", h=num_heads)
        k_bg = rearrange(k_bg, "(b h) n d -> h (b n) d", h=num_heads)
        q_fg = rearrange(q_fg, "(b h) n d -> h (b n) d", h=num_heads)
        k_fg = rearrange(k_fg, "(b h) n d -> h (b n) d", h=num_heads)
        v_bg = rearrange(v_bg, "(b h) n d -> h (b n) d", h=num_heads)
        v_fg = rearrange(v_fg, "(b h) n d -> h (b n) d", h=num_heads)

        mask_t = self.mask_t.unsqueeze(0).unsqueeze(0)
        mask_t = F.interpolate(mask_t, (H, W)).flatten(0).unsqueeze(0)
        mask_t = mask_t.flatten()

        mask_r = self.mask_s.unsqueeze(0).unsqueeze(0)
        mask_r = F.interpolate(mask_r, (H, W)).flatten(0).unsqueeze(0)
        mask_r = mask_r.flatten()
            
        sim_fg = torch.einsum("h i d, h j d -> h i j", q_fg, k_fg) * kwargs.get("scale")
        sim_fg = sim_fg + mask_r.masked_fill(mask_r == 0, torch.finfo(q_bg.dtype).min)

        sim_bg = torch.einsum("h i d, h j d -> h i j", q_bg, k_bg) * kwargs.get("scale")
        sim_bg = sim_bg + mask_t.masked_fill(mask_t == 1, torch.finfo(q_bg.dtype).min)

        attn_fg = sim_fg.softmax(-1)
        attn_bg = sim_bg.softmax(-1)

        out_fg = torch.einsum("h i j, h j d -> h i d", attn_fg, v_fg)
        out_bg = torch.einsum("h i j, h j d -> h i d", attn_bg, v_bg)

        out = torch.cat([out_fg, out_bg])

        out = rearrange(out, "(h1 h) (b n) d -> (h1 b) n (h d)", b=B, h=num_heads)
        
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        B = q.shape[0] // num_heads // 2
        H = W = int(np.sqrt(q.shape[1]))
        
        if is_cross:
            # # amplifies attention to anomalous words.
            # if self.num_cross_replace[0] <= self.cur_step < self.num_cross_replace[1]:
            #     attnu, attnc = attn.chunk(2)
            #     attnc[None, -num_heads:, :, :] = attnc[None, -num_heads:, :, :] * self.equalizer[:, None, None, :]
            #     attn = torch.cat([attnu, attnc], dim=0)
            
            # Amplification is performed only within the set step range and mask area.
            if self.num_cross_replace[0] <= self.cur_step < self.num_cross_replace[1]:
                attnu, attnc = attn.chunk(2, dim=0)
                BH, N, T = attnc.shape
                Hh = num_heads
                B  = BH // Hh            # 3
                block = attnc[-Hh:]      # [H, N, T] == [8, N, T]
                # 2) Column Gain (Amplify only the required tokens; other columns = 1)
                eq = self.equalizer.to(attn.device).to(attn.dtype)   
                eq = eq.view(1, 1, T)
                # 3) Gate mask_s
                r = int(N ** 0.5)
                gate = F.interpolate(self.mask_t[None, None].float().to(attn.device), size=(r, r), mode="nearest")     # [1,1,r,r]
                gate = gate.view(1, N, 1)
                gain = 1.0 + gate * (eq - 1.0)
                block = block * gain
                block = block / block.sum(-1, keepdim=True).clamp_min(1e-6)
                attnc[-Hh:] = block
                attn = torch.cat([attnu, attnc], dim=0)

            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        elif self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]:
            # ours
            # text_embeddings = torch.cat([unconditional_embeddings, text_embeddings]
            qu, qc = q.chunk(2)
            ku, kc = k.chunk(2)
            vu, vc = v.chunk(2)
            attnu, attnc = attn.chunk(2)

            out_u_source = self.attn_batch(qu[:num_heads], ku[:num_heads], vu[:num_heads], sim[:num_heads], attnu, is_cross, place_in_unet, num_heads, **kwargs)
            out_u_condition = self.attn_batch(qu[num_heads:2*num_heads], ku[num_heads:2*num_heads], vu[num_heads:2*num_heads], sim[num_heads:2*num_heads], attnu, is_cross, place_in_unet, num_heads, **kwargs)
            # source-normal; condition-anomaly
            out_c_source = self.attn_batch(qc[:num_heads], kc[:num_heads], vc[:num_heads], sim[:num_heads], attnc, is_cross, place_in_unet, num_heads, **kwargs)
            out_c_condition = self.attn_batch(qc[num_heads:2*num_heads], kc[num_heads:2*num_heads], vc[num_heads:2*num_heads], sim[num_heads:2*num_heads], attnc, is_cross, place_in_unet, num_heads, **kwargs)

            out_u_target = self.attn_batch2(qu[:num_heads], ku[:num_heads], qu[num_heads:2*num_heads], ku[num_heads:2*num_heads], vu[:num_heads], vu[num_heads:2*num_heads], num_heads, **kwargs)
            out_c_target = self.attn_batch2(qc[:num_heads], kc[:num_heads], qc[num_heads:2*num_heads], kc[num_heads:2*num_heads], vc[:num_heads], vc[num_heads:2*num_heads], num_heads, **kwargs)
            
            if self.mask_s is not None and self.mask_t is not None:
                out_u_target_fg, out_u_target_bg = out_u_target.chunk(2, 0)
                out_c_target_fg, out_c_target_bg = out_c_target.chunk(2, 0)

                mask = F.interpolate(self.mask_t.unsqueeze(0).unsqueeze(0), (H, W))
                mask = mask.reshape(-1, 1)  # (hw, 1)
                # mask_r = self.mask_s.to(q.dtype).to(q.device)[None, None, ...]     # [1,1,512,512]
                # mask_r = F.adaptive_max_pool2d(mask_r, (H, W))                          # [1,1,H,W]
                out_u_target = out_u_target_fg * mask + out_u_target_bg * (1 - mask)
                out_c_target = out_c_target_fg * mask + out_c_target_bg * (1 - mask)

            out = torch.cat([out_u_source, out_u_condition, out_u_target, out_c_source, out_c_condition, out_c_target], dim=0)

            return out

        elif self.cur_step in self.step_idx and self.cur_att_layer // 2 in self.layer_idx:
            # print(self.layer_idx) [9, 10, 11, 12, 13, 14, 15]  print(self.step_idx) [4-49]
            # 这个是处理self-attention后几层的，就不跟正常的一样进入attention_store里处理，所以没存到
            if self.attn_store_judge:
                # 我再加个进入attention_store的处理，仅作存储Q\K\V
                super().storage_QKV(q, k, v, sim, attn, is_cross, place_in_unet, num_heads)
            
            qu, qc = q.chunk(2)
            ku, kc = k.chunk(2)
            vu, vc = v.chunk(2)
            attnu, attnc = attn.chunk(2)

            out_u_source = self.attn_batch(qu[:num_heads], ku[:num_heads], vu[:num_heads], sim[:num_heads], attnu, is_cross, place_in_unet, num_heads, **kwargs)
            out_u_condition = self.attn_batch(qu[num_heads:2*num_heads], ku[num_heads:2*num_heads], vu[num_heads:2*num_heads], sim[num_heads:2*num_heads], attnu, is_cross, place_in_unet, num_heads, **kwargs)
            out_u_target = self.attn_batch(qu[-num_heads:], ku[num_heads:2*num_heads], vu[num_heads:2*num_heads], sim[num_heads:2*num_heads], attnu, is_cross, place_in_unet, num_heads, is_mask_attn=True, k_bg=ku[:num_heads], v_bg=vu[:num_heads], is_mask_attn_reweight=True, **kwargs)

            out_c_source = self.attn_batch(qc[:num_heads], kc[:num_heads], vc[:num_heads], sim[:num_heads], attnc, is_cross, place_in_unet, num_heads, **kwargs)
            out_c_condition = self.attn_batch(qc[num_heads:2*num_heads], kc[num_heads:2*num_heads], vc[num_heads:2*num_heads], sim[num_heads:2*num_heads], attnc, is_cross, place_in_unet, num_heads, **kwargs)
            out_c_target = self.attn_batch(qc[-num_heads:], kc[num_heads:2*num_heads], vc[num_heads:2*num_heads], sim[num_heads:2*num_heads], attnc, is_cross, place_in_unet, num_heads, is_mask_attn=True, k_bg=kc[:num_heads], v_bg=vc[:num_heads], is_mask_attn_reweight=True, **kwargs)

            if self.mask_s is not None and self.mask_t is not None:
                out_u_target_fg, out_u_target_bg = out_u_target.chunk(2, 0)
                out_c_target_fg, out_c_target_bg = out_c_target.chunk(2, 0)

                mask = self.mask_t.to(out_c_target.dtype).to(out_c_target.device)[None, None, ...]   # [1,1,512,512]
                mask = F.adaptive_max_pool2d(mask, (H, W))                     # [1,1,H,W]
                mask = mask.view(-1, 1)

                out_u_target = out_u_target_fg * mask + out_u_target_bg * (1 - mask)
                out_c_target = out_c_target_fg * mask + out_c_target_bg * (1 - mask)

                # out_u_target = out_u_target_fg + out_u_target_bg
                # out_c_target = out_c_target_fg + out_c_target_bg

            out = torch.cat([out_u_source, out_u_condition, out_u_target, out_c_source, out_c_condition, out_c_target], dim=0)

            return out

        else:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 32 ** 2:
            return attn_base
        else:
            return att_replace
        