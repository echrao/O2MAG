# from https://github.com/dreamgaussian/dreamgaussian

from diffusers import (
    DDIMScheduler,
    StableDiffusionPipeline,
)
from diffusers.utils.import_utils import is_xformers_available


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class StableDiffusion(nn.Module):
    def __init__(
        self,
        device='cuda',
        fp16=True,
        vram_O=False,
        model_path="runwayml/stable-diffusion-v1-5",
    ):
        super().__init__()

        self.device = device

        self.dtype = torch.float16 if fp16 else torch.float32

        # Create model
        pipe = StableDiffusionPipeline.from_pretrained(
            model_path, torch_dtype=self.dtype
        )

        if vram_O:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            # pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        if is_xformers_available():
            self.unet.enable_xformers_memory_efficient_attention()

        self.scheduler = DDIMScheduler.from_pretrained(
            model_path, subfolder="scheduler", torch_dtype=self.dtype
        )

        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        # following the paper, t in [1, T]
        self.min_step = 1
        self.max_step = self.num_train_timesteps
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience

        self.embeddings = {}

        self.noise = torch.randn((1, 4, 64, 64)).to(self.device, self.dtype) # fix the noise for step A (B and C?)

    def get_embedding(self):
        return self.embeddings['pos'].detach().clone()

    # @torch.no_grad()
    def get_text_embeds(self, prompts, negative_prompts):
        pos_embeds = self.encode_text(prompts).clone().detach()  # [1, 77, 768]
        neg_embeds = self.encode_text(negative_prompts)
        self.embeddings['pos'] = pos_embeds
        self.embeddings['neg'] = neg_embeds
    
    def encode_text(self, prompt):
        # prompt: [str]
        inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]
        return embeddings

    def train_step(
        self,
        pred_rgb,
        guidance_scale=7.5,
        as_latent=False,
    ):
        
        batch_size = pred_rgb.shape[0]
        pred_rgb = pred_rgb.to(self.dtype)

        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode="bilinear", align_corners=False) * 2 - 1
        else:
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode="bilinear", align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)

        t = torch.randint(self.min_step, self.max_step, (batch_size,), dtype=torch.long, device=self.device)

        # w(t), sigma_t^2
        w = (1 - self.alphas[t]).view(batch_size, 1, 1, 1)

        # add noise
        noise = torch.randn_like(latents)
        # noise = self.noise
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        # pred noise
        latent_model_input = torch.cat([latents_noisy] * 2)
        tt = torch.cat([t] * 2)

        embeddings = torch.cat([self.embeddings['pos'].expand(batch_size, -1, -1), self.embeddings['neg'].expand(batch_size, -1, -1)])

        noise_pred = self.unet(
            latent_model_input, tt, encoder_hidden_states=embeddings
        ).sample

        noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_cond - noise_pred_uncond
        )

        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

        return loss

    @torch.no_grad()
    def produce_latents(
        self,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
        prompt_embeds=None,
    ):
        if latents is None:
            # latents = torch.randn(
            #     (
            #         1,
            #         self.unet.in_channels,
            #         height // 8,
            #         width // 8,
            #     ),
            #     device=self.device,
            #     dtype=self.dtype,
            # )
            latents = self.noise

        batch_size = latents.shape[0]
        self.scheduler.set_timesteps(num_inference_steps)
        # embeddings = torch.cat([self.embeddings['pos'].expand(batch_size, -1, -1), self.embeddings['neg'].expand(batch_size, -1, -1)])
        embeddings = torch.cat([prompt_embeds.expand(batch_size, -1, -1), self.embeddings['neg'].expand(batch_size, -1, -1)])

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=embeddings
            ).sample

            # perform guidance
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        return latents

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents

    def prompt_to_img(
        self,
        prompt_embeds,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
    ):
        # Text embeds -> img latents
        latents = self.produce_latents(
            height=height,
            width=width,
            latents=latents,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            prompt_embeds=prompt_embeds
        )  # [1, 4, 64, 64]

        # Img latents -> imgs
        imgs = self.decode_latents(latents)  # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype("uint8")

        return imgs