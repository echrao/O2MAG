# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import Optional, Union, Tuple, List, Callable, Dict
from IPython.display import display
from tqdm.notebook import tqdm
import random
from collections import deque
import os, json, hashlib, time
from typing import Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt

def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    # font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf", font_size)
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y ), font, 1, text_color, 2)
    return img


def view_images(images, num_rows=1, offset_ratio=0.02):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    display(pil_img)

    return pil_img

class ShuffledCycleSampler:
    def __init__(self, items, seed=None):
        assert len(items) > 0
        self.items = list(items)
        self.rng = random.Random(seed)
        self.queue = deque()
        self._reshuffle()

    def _reshuffle(self):
        self.rng.shuffle(self.items)
        self.queue.extend(self.items)

    def next_one(self):
        if not self.queue:
            self._reshuffle()
        return self.queue.popleft()

def expand_mask_tensor(mask_hw: torch.Tensor, radius: int = 3, iters: int = 1) -> torch.Tensor:
    mask = (mask_hw > 0.5).float()                    # 保底二值化
    if radius <= 0 or mask.numel() == 0:
        return mask

    x = mask.unsqueeze(0).unsqueeze(0)               # [1,1,H,W]
    k = 2 * int(radius) + 1
    for _ in range(max(1, int(iters))):
        x = F.max_pool2d(x, kernel_size=k, stride=1, padding=int(radius))
    out = (x.squeeze(0).squeeze(0) > 0.5).float()    # 回到 [H,W] 且二值
    return out

def expand_mask_from_path(source_mask_path: str,
                          size: int = 512,
                          device: str | torch.device = "cuda:0",
                          thresh: float = 0.5,
                          radius: int = 5,
                          iters: int = 1) -> torch.Tensor:
    pil = Image.open(source_mask_path).convert("L")
    pil = pil.resize((size, size), Image.NEAREST)
    m = T.ToTensor()(pil).to(device)                 # [1,H,W], 0~1
    m = (m > thresh).float().squeeze(0)              # [H,W], 0/1
    m = expand_mask_tensor(m, radius=radius, iters=iters)
    return m.to(device)

@torch.no_grad()
def shift_mask_to_point(mask_hw: torch.Tensor,
                        target_xy: tuple,                # (x, y)
                        anchor: str = "bbox",            # "bbox" | "centroid"
                        subpixel: bool = False,          # True 用 grid_sample
                        point_scale_from: tuple | None = (512, 512)  # 默认来自 512×512
                       ) -> torch.Tensor:
    mask = (mask_hw > 0.5).to(mask_hw.dtype)
    H, W = mask.shape
    x_t, y_t = map(float, target_xy)

    # 把 512×512(或其他)坐标缩放到当前大小
    if point_scale_from is not None:
        W0, H0 = map(float, point_scale_from)
        x_t *= (W / max(W0, 1.0))
        y_t *= (H / max(H0, 1.0))

    ys, xs = torch.nonzero(mask > 0.5, as_tuple=True)
    if xs.numel() == 0:
        return torch.zeros_like(mask)

    if anchor == "bbox":
        cx = (xs.min().item() + xs.max().item()) / 2.0
        cy = (ys.min().item() + ys.max().item()) / 2.0
    elif anchor == "centroid":
        cx = xs.float().mean().item()
        cy = ys.float().mean().item()
    else:
        raise ValueError("anchor 只能是 'bbox' 或 'centroid'")

    dx, dy = (x_t - cx), (y_t - cy)

    if subpixel:
        x4d = mask.unsqueeze(0).unsqueeze(0)             # [1,1,H,W]
        tx = 2.0 * dx / max(W - 1, 1)
        ty = 2.0 * dy / max(H - 1, 1)
        theta = mask.new_tensor([[1, 0, tx], [0, 1, ty]]).unsqueeze(0)
        grid = F.affine_grid(theta, size=(1,1,H,W), align_corners=False)
        moved = F.grid_sample(x4d, grid, mode='nearest',
                              padding_mode='zeros', align_corners=False).squeeze()
        return (moved > 0.5).to(mask_hw.dtype)
    else:
        dx_i, dy_i = int(round(dx)), int(round(dy))
        out = torch.zeros_like(mask)
        ys2, xs2 = ys + dy_i, xs + dx_i
        m = (xs2 >= 0) & (ys2 >= 0) & (xs2 < W) & (ys2 < H)
        out[ys2[m], xs2[m]] = 1
        return out.to(mask_hw.dtype)

class MVTecBankSimple:
    def __init__(self, bank_path="./mvtec_embed_bank"):
        self.root = bank_path
        os.makedirs(self.root, exist_ok=True)

    def _path(self, category: str, defect: str, image_path: str) -> str:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        out_dir = os.path.join(self.root, category, defect)
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, f"{stem}.pt")

    def exists(self, category, defect, image_path) -> bool:
        return os.path.exists(self._path(category, defect, image_path))

    def load(self, category, defect, image_path, map_location="cpu"):
        p = self._path(category, defect, image_path)
        if not os.path.exists(p): return None
        obj = torch.load(p, map_location=map_location)
        return obj["embed"]  # Tensor(CPU)

    def save(self, category, defect, image_path, emb):
        p = self._path(category, defect, image_path)
        tmp = p + ".tmp"
        torch.save({"embed": emb.detach().to("cpu"), "ts": time.time()}, tmp)
        os.replace(tmp, p)
        return p

def show_overlay_pil(img_pil: Image.Image,
                     mask_tensor: torch.Tensor,
                     title: str = "RESULT IMAGE",
                     mask_color=(0, 255, 0),
                     alpha: float = 0.35,
                     save_overlay_path: str | None = None,   # 新增：保存合成后的单图
                     save_figure_path: str | None = None,    # 新增：保存左右对比大图
                     figure_dpi: int = 150) -> Image.Image:  # 新增：返回合成图
    img_pil = img_pil.convert("RGB")
    W, H = img_pil.size
    img_np = np.array(img_pil, dtype=np.uint8)

    m = mask_tensor.detach().float().cpu()
    if m.ndim == 4:
        m = m.squeeze(0).squeeze(0)
    elif m.ndim == 3:
        if m.shape[0] == 1: m = m.squeeze(0)
        elif m.shape[-1] == 1: m = m.squeeze(-1)
    assert m.ndim == 2, f"mask_tensor 期望是 2D，得到形状 {tuple(m.shape)}"
    m_bool = (m > 0.5)

    h, w = m_bool.shape
    if (w, h) != (W, H):
        m_pil = Image.fromarray(m_bool.numpy().astype(np.uint8) * 255)
        m_pil = m_pil.resize((W, H), Image.NEAREST)
        m_bool = np.array(m_pil) > 0
    else:
        m_bool = m_bool.numpy()

    overlay = np.zeros((H, W, 4), dtype=np.uint8)
    overlay[..., :3] = np.array(mask_color, dtype=np.uint8)
    overlay[..., 3]  = (m_bool.astype(np.uint8) * int(round(alpha * 255)))

    base_rgba = Image.fromarray(img_np).convert("RGBA")
    over_rgba = Image.fromarray(overlay, mode="RGBA")
    comp = Image.alpha_composite(base_rgba, over_rgba)  # RGBA

    if save_overlay_path is not None:
        ext = str(save_overlay_path).lower()
        if ext.endswith(".jpg") or ext.endswith(".jpeg"):
            comp.convert("RGB").save(save_overlay_path, quality=95)
        else:
            comp.save(save_overlay_path)  # PNG 保留透明通道

    fig, ax = plt.subplots(1, 2, figsize=(8, 4), dpi=figure_dpi)
    ax[0].imshow(img_np); ax[0].set_title(f"{title} • image");   ax[0].axis("off")
    ax[1].imshow(comp);   ax[1].set_title(f"{title} • overlay"); ax[1].axis("off")
    plt.tight_layout()

    if save_figure_path is not None:
        fig.savefig(save_figure_path, bbox_inches="tight")
    plt.show()

    return fig


# compute attn
def report_row_sums(attn_bht: torch.Tensor, num_heads: int, tag: str):
    """
    attn_bht: [B*H, N, T] 的注意力（通常是 cond 半边 attnc）
    沿 token 维求和 -> [B*H, N]，再按样本分块，打印每个样本 |sum-1| 的统计
    """
    BH, N, T = attn_bht.shape
    B = BH // num_heads
    sums = attn_bht.sum(dim=-1)                    # [BH, N]
    err  = (sums - 1.0).abs()
    print(f"[{tag}] shape={list(attn_bht.shape)}, "
          f"global max|sum-1|={err.max().item():.6e}, "
          f"mean|sum-1|={err.mean().item():.6e}")

    sums_b = sums.view(B, num_heads, N)            # [B, H, N]
    for b in range(B):
        e = (sums_b[b] - 1.0).abs()
        print(f"  sample {b}: max|sum-1|={e.max().item():.6e}, "
              f"mean|sum-1|={e.mean().item():.6e}")

def diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False):
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents = controller.step_callback(latents)
    return latents


def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        latent = torch.randn(
            (1, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.expand(batch_size,  model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents


@torch.no_grad()
def text2image_ldm(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
):
    register_attention_control(model, controller)
    height = width = 256
    batch_size = len(prompt)
    
    uncond_input = model.tokenizer([""] * batch_size, padding="max_length", max_length=77, return_tensors="pt")
    uncond_embeddings = model.bert(uncond_input.input_ids.to(model.device))[0]
    
    text_input = model.tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt")
    text_embeddings = model.bert(text_input.input_ids.to(model.device))[0]
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    context = torch.cat([uncond_embeddings, text_embeddings])
    
    model.scheduler.set_timesteps(num_inference_steps)
    for t in tqdm(model.scheduler.timesteps):
        latents = diffusion_step(model, controller, latents, context, t, guidance_scale)
    
    image = latent2image(model.vqvae, latents)
   
    return image, latent


@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt: List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    low_resource: bool = False,
):
    register_attention_control(model, controller)
    height = width = 512
    batch_size = len(prompt)

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    
    context = [uncond_embeddings, text_embeddings]
    if not low_resource:
        context = torch.cat(context)
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    
    # set timesteps
    extra_set_kwargs = {"offset": 1}
    model.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)
    for t in tqdm(model.scheduler.timesteps):
        latents = diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource)
    
    image = latent2image(model.vae, latents)
  
    return image, latent


def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, context=None, mask=None):
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q = self.reshape_heads_to_batch_dim(q)
            k = self.reshape_heads_to_batch_dim(k)
            v = self.reshape_heads_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if mask is not None:
                mask = mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            attn = controller(attn, is_cross, place_in_unet)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.reshape_batch_dim_to_heads(out)
            return to_out(out)

        return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count

    
def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


def update_alpha_time_word(alpha, bounds: Union[float, Tuple[float, float]], prompt_ind: int,
                           word_inds: Optional[torch.Tensor]=None):
    if type(bounds) is float:
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[: start, prompt_ind, word_inds] = 0
    alpha[start: end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha


def get_time_words_attention_alpha(prompts, num_steps,
                                   cross_replace_steps: Union[float, Dict[str, Tuple[float, float]]],
                                   tokenizer, max_num_words=77):
    if type(cross_replace_steps) is not dict:
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0., 1.)
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 2, max_num_words)
    for i in range(len(prompts) - 2):
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"],
                                                  i)
    for key, item in cross_replace_steps.items():
        if key != "default_":
             inds = [get_word_inds(prompts[i], key, tokenizer) for i in range(2, len(prompts))]
             for i, ind in enumerate(inds):
                 if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 2, 1, 1, max_num_words)
    return alpha_time_words

def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
                  Tuple[float, ...]], tokenizer):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(len(values), 77)
    values = torch.tensor(values, dtype=torch.float32)
    for word in word_select:
        inds = get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = values
    return equalizer