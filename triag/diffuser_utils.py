"""
Util functions based on Diffuser framework.
"""


import os
import torch
import cv2
import numpy as np

import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
from PIL import Image
from torchvision.utils import save_image
from torchvision.io import read_image
from torchvision.transforms.functional import pil_to_tensor
from tqdm import trange

from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from diffusers.utils import load_image

from pytorch_lightning import seed_everything

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from collections import defaultdict

class LocalBlend:

    def __call__(self, x_t, source):
        _,_,H,W = x_t.shape

        mask = self.mask.unsqueeze(0).unsqueeze(0)
        mask = F.interpolate(mask, (H, W))

        x_t = (1-mask) * source + mask * x_t

        return x_t
       
    def __init__(self, mask):
        self.mask = mask

class McaPipeline_Replace(StableDiffusionPipeline):
    
    def get_sg_aux(self, cfg=True, transpose=True):
        aux = defaultdict(dict)
        for name, aux_module in self.unet.named_modules():
            try:
                module_aux = aux_module._aux
                if transpose:
                    for k, v in module_aux.items():
                        if cfg: 
                            v = torch.utils._pytree.tree_map(lambda vv: vv.chunk(2)[1] if vv is not None else None, v)
                        aux[k][name] = v
                else:
                    aux[name] = module_aux
                    if cfg:
                        aux[name] = {k: torch.utils._pytree.tree_map(lambda vv: vv.chunk(2)[1] if vv is not None else None, v) for k, v in aux[name].items()}
            except AttributeError: 
                pass
        return aux

    def next_step(
            self,
            model_output: torch.FloatTensor,   # ε̂(t) - UNet 预测的噪声
            timestep: int,
            x: torch.FloatTensor,
            eta=0.,
            verbose=False
        ):
        """
        Inverse sampling for DDIM Inversion

        Returns
        -------
        x_next : torch.FloatTensor   # 即 x_{t-1}
        pred_x0: torch.FloatTensor   # 估算的重建 x_0
        """
        if verbose:
            print("timestep: ", timestep)
        next_step = timestep
        timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999)
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_step]
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_next)**0.5 * model_output
        x_next = alpha_prod_t_next**0.5 * pred_x0 + pred_dir
        return x_next, pred_x0

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
        eta: float=0.0,
        verbose=False,
        ):
        """
        predict the sampe the next step in the denoise process.
        """
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep > 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_prev)**0.5 * model_output
        x_prev = alpha_prod_t_prev**0.5 * pred_x0 + pred_dir
        return x_prev, pred_x0

    @torch.no_grad()
    def image2latent(self, image):
        DEVICE = (getattr(self, "_execution_device", None)
        or getattr(self, "device", None)
        or next(self.unet.parameters()).device)
        # DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if type(image) is Image:
            image = np.array(image)
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        # input image density range [-1, 1]
        latents = self.vae.encode(image)['latent_dist'].mean
        latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        elif return_type == "pt":
            image = (image / 2 + 0.5).clamp(0, 1)

        return image

    def latent2image_grad(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents)['sample']

        return image  # range [-1, 1]

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]
        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        batch_size=1,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        eta=0.0,
        latents=None,
        unconditioning=None,
        neg_prompt=None,
        ref_intermediate_latents=None,
        return_intermediates=False,
        opt_embeddings=None,
        lbl = None,
        **kwds):

        # DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        DEVICE = (getattr(self, "_execution_device", None)
        or getattr(self, "device", None)
        or next(self.unet.parameters()).device)
        
        # 初始化下我们想要优化的embedding
        self.embeddings = {}

        if isinstance(prompt, list):
            batch_size = len(prompt)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size

        # text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )

        text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
        print("input text embeddings :", text_embeddings.shape)  # torch.Size([3, 77, 768])
        if kwds.get("dir"):
            dir = text_embeddings[-2] - text_embeddings[-1]
            u, s, v = torch.pca_lowrank(dir.transpose(-1, -2), q=1, center=True)
            text_embeddings[-1] = text_embeddings[-1] + kwds.get("dir") * v

        # define initial latents
        latents_shape = (batch_size, self.unet.in_channels, height//8, width//8)
        if latents is None:
            latents = torch.randn(latents_shape, device=DEVICE)
        else:
            assert latents.shape == latents_shape, f"The shape of input latent tensor {latents.shape} should equal to predefined one."

        # unconditional embedding for classifier free guidance
        if guidance_scale > 1.:
            max_length = text_input.input_ids.shape[-1]
            if neg_prompt:
                uc_text = neg_prompt
            else:
                uc_text = ""
            # uc_text = "ugly, tiling, poorly drawn hands, poorly drawn feet, body out of frame, cut off, low contrast, underexposed, distorted face"
            unconditional_input = self.tokenizer(
                [uc_text] * batch_size,
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            )

            # unconditional_input.input_ids = unconditional_input.input_ids[:, 1:]
            unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(DEVICE))[0]

            if opt_embeddings is not None:
                text_embeddings[-1:] = opt_embeddings

            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)

        print("latents shape: ", latents.shape)
        # iterative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        # print("Valid timesteps: ", reversed(self.scheduler.timesteps))

        latents_list = [latents]
        pred_x0_list = [latents]
        cross_image = []
        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="DDIM Sampler")):

            if ref_intermediate_latents is not None:
                latents_ref = ref_intermediate_latents[0][-1 - i]
                latents_source = ref_intermediate_latents[-1][-1 - i]
                _, _, latents_cur = latents.chunk(3)
                latents = torch.cat([latents_source, latents_ref, latents_cur])

            if guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents
            if unconditioning is not None and isinstance(unconditioning, list):
                _, text_embeddings = text_embeddings.chunk(2)
                text_embeddings = torch.cat([unconditioning[i].expand(*text_embeddings.shape), text_embeddings]) 
            
            # # predict the noise
            # from thop import profile, clever_format
            # flops, params = profile(self.unet, (model_inputs, t, text_embeddings,))
            # print('Flops: % .4fG'%(flops / 1000000000))
            # print('params参数量: % .4fM'% (params / 1000000))
            
            noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings).sample
            # noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings, return_dict=False)[0]
            if guidance_scale > 1.:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
        
            # compute the previous noise sample x_t -> x_t-1
            # latents.detach()
            latents, pred_x0 = self.step(noise_pred, t, latents)

            if lbl is not None:
                latent_source, latent_ref, latent_cur = latents.chunk(3)
                latent_cur = lbl(latent_cur, ref_intermediate_latents[-1][-1-i])
                latents = torch.cat([latent_source, latent_ref, latent_cur])
            latents_list.append(latents)
            pred_x0_list.append(pred_x0)

            if i in [5,10,15,20,25,30,35,40,45]:
                cross_image.append(self.latent2image(latents, return_type="pt"))

        print("latents shape: ", latents.shape)
        mask_latent = self.latent_pair_mask(latents, idx_normal=0, idx_anom=2, q=0.80, smooth_ks=3, area_ks=7, min_count=8)
        mask_up = F.interpolate(mask_latent[None,None], scale_factor=8, mode="nearest")[0,0]

        image = self.latent2image(latents, return_type="pt")
        cross_image.append(image)
        if return_intermediates:
            # pred_x0_list = [self.latent2image(img, return_type="pt") for img in pred_x0_list]
            # latents_list = [self.latent2image(img, return_type="pt") for img in latents_list]
            return image, cross_image
        return image

    @torch.no_grad()
    def invert(
        self,
        image: torch.Tensor,
        prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
        eta=0.0,
        return_intermediates=False,
        **kwds):
        """
        invert a real image into noise map with determinisc DDIM inversion
        """
        # DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        DEVICE = (getattr(self, "_execution_device", None)
              or getattr(self, "device", None)
              or next(self.unet.parameters()).device)
        batch_size = image.shape[0]
        if isinstance(prompt, list):
            if batch_size == 1:
                image = image.expand(len(prompt), -1, -1, -1)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size

        # text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
        # print("input text embeddings :", text_embeddings.shape)

        # define initial latents
        latents = self.image2latent(image)
        start_latents = latents
        # print(latents)
        # exit()

        # unconditional embedding for classifier free guidance
        if guidance_scale > 1.:
            max_length = text_input.input_ids.shape[-1]
            unconditional_input = self.tokenizer(
                [""] * batch_size,
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            )
            unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(DEVICE))[0]
            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)

        # print("latents shape: ", latents.shape)
        # interative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        # print("Valid timesteps: ", reversed(self.scheduler.timesteps))
        # print("attributes: ", self.scheduler.__dict__)
        latents_list = [latents]
        pred_x0_list = [latents]
        for i, t in enumerate(tqdm(reversed(self.scheduler.timesteps), desc="DDIM Inversion")):
            if guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents

            # predict the noise
            noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings).sample
            # noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings)[1]
            if guidance_scale > 1.:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
            # compute the previous noise sample x_t-1 -> x_t
            latents, pred_x0 = self.next_step(noise_pred, t, latents)
            latents_list.append(latents)
            pred_x0_list.append(pred_x0)

        if return_intermediates:
            # return the intermediate laters during inversion
            # 按序保存 pred_x0_list 能看到原图从清晰→逐渐染噪的过程
            # pred_x0_list = [self.latent2image(img, return_type="pt") for img in pred_x0_list]
            return latents, latents_list
        return latents, start_latents
    
    @torch.no_grad()
    def latent_pair_mask(
        self,
        latents: torch.Tensor, 
        idx_normal: int = 0, 
        idx_anom: int = 2,
        q: float = 0.97,
        smooth_ks: int = 3,
        area_ks: int = 7,
        min_count: int = 8
    ):
        """
        latents: [B, C, H, W]，例如 [3,4,64,64]
        返回: mask_latent [H, W]，float{0,1}
        """
        assert latents.dim() == 4, f"expect [B,C,H,W], got {latents.shape}"
        B, C, H, W = latents.shape
        assert idx_normal < B and idx_anom < B, f"indices out of range: {idx_normal}, {idx_anom}, B={B}"

        A = latents[idx_normal]   # [C,H,W]
        B_ = latents[idx_anom]    # [C,H,W]

        # --- 1) 构造差异热图（两种度量，合并更稳） ---
        # L2 差异（通道维）
        l2_map = (A - B_).pow(2).sum(dim=0).sqrt()                  # [H,W]
        # 余弦“距离”
        cos_sim = F.cosine_similarity(A, B_, dim=0).clamp(-1, 1)    # [-1,1] -> [H,W]
        cos_dist = 1.0 - cos_sim                                    # [0,2]，越大越不相似

        # 标准化后相加（z-score 融合，更鲁棒）
        def zscore(x):
            m, s = x.mean(), x.std().clamp(min=1e-6)
            return (x - m) / s
        score = zscore(l2_map) + zscore(cos_dist)                   # [H,W]
        score = (score - score.min()) / (score.max() - score.min() + 1e-6)

        # --- 2) 自适应阈值（分位数） ---
        thr = torch.quantile(score.view(-1), q)
        mask = (score >= thr).float()                                # [H,W]

        # --- 3) 形态学清理（可选，全用 PyTorch 实现） ---
        def bin_dilate(x, k):
            if k <= 1: return x
            pad = k // 2
            ker = torch.ones(1,1,k,k, device=x.device, dtype=x.dtype)
            y = F.conv2d(x[None,None], ker, padding=pad) > 0
            return y.float()[0,0]

        def bin_erode(x, k):
            if k <= 1: return x
            pad = k // 2
            ker = torch.ones(1,1,k,k, device=x.device, dtype=x.dtype)
            y = F.conv2d(x[None,None], ker, padding=pad) >= (k*k)
            return y.float()[0,0]

        if smooth_ks and smooth_ks >= 3 and smooth_ks % 2 == 1:
            # 开运算：去小噪声
            mask = bin_erode(mask, smooth_ks)
            mask = bin_dilate(mask, smooth_ks)
            # 闭运算：填小孔
            mask = bin_dilate(mask, smooth_ks)
            mask = bin_erode(mask, smooth_ks)

        # 小团块抑制（近似的“面积滤波”）
        if area_ks and area_ks >= 3 and area_ks % 2 == 1 and min_count > 0:
            pad = area_ks // 2
            ker = torch.ones(1,1,area_ks,area_ks, device=mask.device, dtype=mask.dtype)
            cnt = F.conv2d(mask[None,None], ker, padding=pad)[0,0]   # 每个像素邻域内的 1 的个数
            mask = (mask * (cnt >= min_count).float())               # 小面积去掉

        return mask  # [H,W]，0/1

def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image

class McaPipeline_generation(StableDiffusionPipeline):

    def next_step(
            self,
            model_output: torch.FloatTensor,
            timestep: int,
            x: torch.FloatTensor,
            eta=0.,
            verbose=False
        ):
        """
        Inverse sampling for DDIM Inversion
        """
        if verbose:
            print("timestep: ", timestep)
        next_step = timestep
        timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999)
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_step]
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_next)**0.5 * model_output
        x_next = alpha_prod_t_next**0.5 * pred_x0 + pred_dir
        return x_next, pred_x0

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
        eta: float=0.0,
        verbose=False,
        ):
        """
        predict the sampe the next step in the denoise process.
        """
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep > 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_prev)**0.5 * model_output
        x_prev = alpha_prod_t_prev**0.5 * pred_x0 + pred_dir
        return x_prev, pred_x0

    @torch.no_grad()
    def image2latent(self, image):
        DEVICE = (getattr(self, "_execution_device", None)
        or getattr(self, "device", None)
        or next(self.unet.parameters()).device)
        # DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if type(image) is Image:
            image = np.array(image)
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        # input image density range [-1, 1]
        latents = self.vae.encode(image)['latent_dist'].mean
        latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        elif return_type == "pt":
            image = (image / 2 + 0.5).clamp(0, 1)

        return image

    def latent2image_grad(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents)['sample']

        return image  # range [-1, 1]

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        batch_size=1,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        eta=0.0,
        latents=None,
        unconditioning=None,
        neg_prompt=None,
        ref_intermediate_latents=None,
        return_intermediates=False,
        mask_r = None,
        mask_t = None,
        **kwds):

        DEVICE = (getattr(self, "_execution_device", None)
        or getattr(self, "device", None)
        or next(self.unet.parameters()).device)
        # DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if isinstance(prompt, list):
            batch_size = len(prompt)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size

        # text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )

        text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
        print("input text embeddings :", text_embeddings.shape)
        if kwds.get("dir"):
            dir = text_embeddings[-2] - text_embeddings[-1]
            u, s, v = torch.pca_lowrank(dir.transpose(-1, -2), q=1, center=True)
            text_embeddings[-1] = text_embeddings[-1] + kwds.get("dir") * v

        # define initial latents
        latents_shape = (batch_size, self.unet.in_channels, height//8, width//8)
        randn_latent = torch.randn(latents_shape, device=DEVICE)
        if latents is None:
            latents = randn_latent
        else:
            # latents[:1] = randn_latent[:1]
            assert latents.shape == latents_shape, f"The shape of input latent tensor {latents.shape} should equal to predefined one."

        # unconditional embedding for classifier free guidance
        if guidance_scale > 1.:
            max_length = text_input.input_ids.shape[-1]
            if neg_prompt:
                uc_text = neg_prompt
            else:
                uc_text = ""
            # uc_text = "ugly, tiling, poorly drawn hands, poorly drawn feet, body out of frame, cut off, low contrast, underexposed, distorted face"
            unconditional_input = self.tokenizer(
                [uc_text] * batch_size,
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            )
            # unconditional_input.input_ids = unconditional_input.input_ids[:, 1:]
            unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(DEVICE))[0]
            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)

        # print("latents shape: ", latents.shape)
        # iterative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        # print("Valid timesteps: ", reversed(self.scheduler.timesteps))

        latents_list = [latents]
        pred_x0_list = [latents]
        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="DDIM Sampler")):

            if ref_intermediate_latents is not None:
                latents_ref = ref_intermediate_latents[-1 - i]
                # latents_source, _, latents_cur = latents.chunk(3)
                # latents = torch.cat([latents_source, latents_ref, latents_cur])
                _, latents_cur = latents.chunk(2)
                latents = torch.cat([latents_ref, latents_cur])

            if guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents
            if unconditioning is not None and isinstance(unconditioning, list):
                _, text_embeddings = text_embeddings.chunk(2)
                text_embeddings = torch.cat([unconditioning[i].expand(*text_embeddings.shape), text_embeddings]) 
            
            # predict the noise
            noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings).sample
            # noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings, return_dict=False)[0]
            if guidance_scale > 1.:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
        
            # compute the previous noise sample x_t -> x_t-1
            # latents.detach()
            latents, pred_x0 = self.step(noise_pred, t, latents)

            latents_list.append(latents)
            pred_x0_list.append(pred_x0)
        
        image = self.latent2image(latents, return_type="pt")
        if return_intermediates:
            # pred_x0_list = [self.latent2image(img, return_type="pt") for img in pred_x0_list]
            # latents_list = [self.latent2image(img, return_type="pt") for img in latents_list]
            return image, randn_latent
        return image

    @torch.no_grad()
    def invert(
        self,
        image: torch.Tensor,
        prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
        eta=0.0,
        return_intermediates=False,
        **kwds):
        """
        invert a real image into noise map with determinisc DDIM inversion
        """
        DEVICE = (getattr(self, "_execution_device", None)
        or getattr(self, "device", None)
        or next(self.unet.parameters()).device)
        # DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        batch_size = image.shape[0]
        if isinstance(prompt, list):
            if batch_size == 1:
                image = image.expand(len(prompt), -1, -1, -1)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size

        # text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
        print("input text embeddings :", text_embeddings.shape)

        # define initial latents
        latents = self.image2latent(image)
        start_latents = latents
        # print(latents)
        # exit()

        # unconditional embedding for classifier free guidance
        if guidance_scale > 1.:
            max_length = text_input.input_ids.shape[-1]
            unconditional_input = self.tokenizer(
                [""] * batch_size,
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            )
            unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(DEVICE))[0]
            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)

        print("latents shape: ", latents.shape)
        # interative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        print("Valid timesteps: ", reversed(self.scheduler.timesteps))
        # print("attributes: ", self.scheduler.__dict__)
        latents_list = [latents]
        pred_x0_list = [latents]
        for i, t in enumerate(tqdm(reversed(self.scheduler.timesteps), desc="DDIM Inversion")):
            if guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents

            # predict the noise
            noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings).sample
            if guidance_scale > 1.:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
            # compute the previous noise sample x_t-1 -> x_t
            latents, pred_x0 = self.next_step(noise_pred, t, latents)
            # print("latents 的形状大小: ", latents.shape)
            latents_list.append(latents)
            pred_x0_list.append(pred_x0)

        if return_intermediates:
            # return the intermediate laters during inversion
            # pred_x0_list = [self.latent2image(img, return_type="pt") for img in pred_x0_list]
            return latents, latents_list
        return latents, start_latents
