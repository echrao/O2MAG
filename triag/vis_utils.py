import math
from typing import List
from PIL import Image
import cv2
import numpy as np
import torch
import os
from einops import rearrange

from triag import ptp_utils
from triag.mca_utils import AttentionStore
from triag.mca_p2p import McaControlReplace
import matplotlib.pyplot as plt

@torch.no_grad()
def q_to_rgb(q_mean: torch.Tensor, save_path: str = None, type_attention = 'Q', place = 0, layer_idx = 0) -> Image.Image:
    """
    q_mean: [B, N, D] —— already averaged head-to-head; optional B=1
    Take the 0th sample, perform SVD to project it into 3D, and then map it to RGB.
    """
    for B in range(len(q_mean)):
        x = q_mean[B]                     # [N, D]
        x = x - x.mean(dim=0, keepdim=True)
        # SVD: x ≈ U @ S @ Vh
        Vh = torch.linalg.svd(x, full_matrices=False).Vh      # [D, D]
        proj = x @ Vh[:3].T                                    # [N, 3]
        N, D = x.shape
        r = int(math.sqrt(N))
        res = r
        # normalize to [0,255]
        proj = (proj - proj.min(0, keepdim=True)[0]) / (proj.max(0, keepdim=True)[0] - proj.min(0, keepdim=True)[0] + 1e-8)
        rgb = (proj.reshape(res, res, 3).cpu().numpy() * 255).astype(np.uint8)
        img = Image.fromarray(rgb)
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            if type_attention=='Q':
                img.save(save_path+f"Q_{place}_Layer{layer_idx}_num{B}.png")
            else:
                img.save(save_path+f"V_{place}_Layer{layer_idx}_num{B}.png")
    return img

@torch.no_grad()
def self_attn_pca_rgb_pnp(
    attn: torch.Tensor,               # (B*H, N, N) or (B, H, N, N)
    grid_shape: tuple|None = None,
    batch_size: int = 3,              # The number of samples forward in this iteration (len(prompts))
    center: str = "col",
):
    if attn.dim() == 4:
        B, Hh, N, M = attn.shape
    elif attn.dim() == 3:
        BH, N, M = attn.shape
        assert batch_size > 0 and BH % batch_size == 0, f"无法从 BH={BH} 拆出 batch_size={batch_size}"
        Hh = BH // batch_size
        B = batch_size
        attn = attn.view(B, Hh, N, M)
    else:
        raise ValueError(f"Unsupported attn shape: {attn.shape}")
    assert N == M, f"self-attn 应该方阵，拿到了 {N}x{M}（可能混入了 cross-attn）"

    # ---- 网格大小 ----
    if grid_shape is None:
        r = int(math.isqrt(N)); assert r*r == N, f"N={N} 非平方，传 grid_shape"
        H, W = r, r
    else:
        H, W = grid_shape
        assert H*W == N, f"grid={H}x{W} 与 N={N} 不匹配"

    outs = []
    onesN = None
    for b in range(B):
        # A: (Hh, N, N)
        A = attn[b].to('cpu', dtype=torch.float32)
        C = A.mean(dim=1)  # (Hh, N)

        if onesN is None:
            onesN = torch.ones(N, 1, dtype=torch.float32)
            J = onesN @ onesN.t()
        G = torch.zeros(N, N, dtype=torch.float32)
        for h in range(A.shape[0]):
            Ah = A[h]          # (N,N)
            ch = C[h]          # (N,)

            # AhAhT
            AhAhT = Ah @ Ah.t()                       # (N,N)
            # v = Ah @ ch  （N,)
            v = Ah @ ch
            # (A c) 1^T  和  1 (A c)^T
            V1 = v.view(-1,1) @ onesN.t()             # (N,N)
            V2 = onesN @ v.view(1,-1)                 # (N,N)
            # (c^T c) 1 1^T
            c2J = torch.dot(ch, ch) * J               # (N,N)

            G += AhAhT - V1 - V2 + c2J

        # 特征分解（对称矩阵）：G = U Λ U^T
        eigvals, eigvecs = torch.linalg.eigh(G)       # 升序
        U3 = eigvecs[:, -3:]                          # (N,3)
        S3 = eigvals[-3:].clamp_min(0).sqrt()         # (3,)
        Z  = U3 * S3                                   # (N,3)  == U*S（PCA 的样本得分）

        # 归一化到 [0,1]（逐通道）
        zmin = Z.min(0, keepdim=True).values
        zmax = Z.max(0, keepdim=True).values
        Z = (Z - zmin) / (zmax - zmin + 1e-6)

        # 还原成 (H,W,3) 并转 uint8
        rgb = (Z.view(H, W, 3).numpy() * 255).astype(np.uint8)
        outs.append(rgb)

    return outs


def aggregate_attention(attention_store: McaControlReplace,
                        res: int,   #  res ** 2 filter
                        is_cross: bool,  # True: cross-attention, False: self-attention
                        select: int) -> torch.Tensor:   # Selecting from multiple samples (batch)
    """ Aggregates the attention across the different layers and heads at the specified resolution. """
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    if is_cross:
        attention_maps = attention_maps[1]
    else:
        attention_maps = attention_maps[0]
    for item in attention_maps:
        if item.shape[1] == num_pixels:
            cross_maps = item.reshape(1, -1, res, res, item.shape[-1])[select]
            out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out

def show_cross_attention(prompt: str,
                         attention_store: McaControlReplace,
                         tokenizer,
                         indices_to_alter: List[int],   # Which tokens' attention do you want to see?
                         res: int,
                         select: int = 0,
                         orig_image=None):
    tokens = tokenizer.encode(prompt)
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, res, True, select).detach().cpu()
    images = []

    # show spatial attention for indices of tokens to strengthen
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        if i in indices_to_alter:
            image = show_image_relevance(image, orig_image)
            image = image.astype(np.uint8)
            image = np.array(Image.fromarray(image).resize((res ** 2, res ** 2)))
            image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
            images.append(image)

    save_img = ptp_utils.view_images(np.stack(images, axis=0))
    save_img.save("./attention_map/cross_attention/attentionmap.png")
    # TODO
    return images

# TODO Already changed to a cross-attention visualization that separates uncodition and condition.
def show_cross_attention_for_list(prompt: str,
                         attention_store: McaControlReplace,
                         tokenizer,
                         indices_to_alter: List[int],
                         res: int,
                         select: int = 0,
                         cross_show_time=[50],
                         batch_size=3):
    tokens = tokenizer.encode(prompt)
    print("正在可视化{}的注意力热图".format(prompt))
    decoder = tokenizer.decode
    attention_maps_list = attention_store.get_average_attention_list()
    num_pixels = res ** 2
    for num,attention_maps in enumerate(attention_maps_list):
        out = []
        images = []
        for item in attention_maps:
            if item.shape[1] == num_pixels:
                # cross_maps = item.reshape(1, -1, res, res, item.shape[-1])[select]
                # num_heads_conwithnocon = item.shape[0] // batch_size
                # cross_maps = item.view(batch_size, num_heads_conwithnocon, res, res, item.shape[-1])[select]
                cross_maps = item.reshape(3, -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
        out = torch.cat(out, dim=0)
        out = out.sum(0) / out.shape[0]
        out = out.detach().cpu()
        attention_maps_final = out

        # Do not display the original image
        for i in range(len(tokens)):
            image = attention_maps_final[:, :, i]
            image = 255 * image / image.max()
            image = image.unsqueeze(-1).expand(*image.shape, 3)
            image = image.numpy().astype(np.uint8)
            image = np.array(Image.fromarray(image).resize((256, 256)))
            image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
            images.append(image)

        save_img = ptp_utils.view_images(np.stack(images, axis=0))
        save_img.save(f"./attention_map/cross_attention/timestep_{cross_show_time[num]}.png")

# TODO It hasn't been modified to separate the cross-attention visualization of uncodition and condition yet.
def show_cross_attention_photo_for_list(prompt: str,
                         attention_store: McaControlReplace,
                         tokenizer,
                         indices_to_alter: List[int],
                         res: int,
                         select: int = 0,
                         orig_image=None,
                         cross_show_time=[50],
                         batch_size=3):
    tokens = tokenizer.encode(prompt)
    decoder = tokenizer.decode
    attention_maps_list = attention_store.get_average_attention_list()
    num_pixels = res ** 2
    for num,attention_maps in enumerate(attention_maps_list):
        out = []
        images = []
        for item in attention_maps:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(3, -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
        out = torch.cat(out, dim=0)
        out = out.sum(0) / out.shape[0]
        out = out.detach().cpu()
        attention_maps = out

        # show spatial attention for indices of tokens to strengthen
        for i in range(len(tokens)):
            image = attention_maps[:, :, i]
            if i in indices_to_alter:
                image = show_image_relevance(image, orig_image)
                image = image.astype(np.uint8)
                image = np.array(Image.fromarray(image).resize((res ** 2, res ** 2)))
                image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
                images.append(image)

        save_img = ptp_utils.view_images(np.stack(images, axis=0))
        save_img.save(f"./attention_map/cross_attention/timestep_{cross_show_time[num]}.png")


# prompt-to-prompt to visualize self-attention
def show_self_attention_comp(attention_store: AttentionStore, res: int, from_where: List[str],
                        max_com=10, select: int = 0):
    attention_maps = aggregate_attention(attention_store, res, from_where, False, select).numpy().reshape((res ** 2, res ** 2))
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    ptp_utils.view_images(np.concatenate(images, axis=1))

def save_attention(prompt: str,
                    attention_maps,
                    tokenizer,
                    res: int,
                    from_where: List[str],
                    select: int = 0,
                    save_dir: str = "attention_vis",
                    t: int = 0,
                    photo_num: int = 0):
    tokens = tokenizer.encode(prompt)
    decoder = tokenizer.decode
    # attention_maps = aggregate_attention(attention_store, res, from_where, True, select)
    # Do not display the original image
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i].detach().cpu()
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
        images.append(image)

        # color photo
        # attention = attention_maps[:, :, i]
        # attention = (attention - attention.min()) / (attention.max() - attention.min())
        # cmap = plt.get_cmap('viridis')
        # colored_attention = (cmap(attention.numpy()) * 255).astype(np.uint8)[:, :, :3]
        # # Adjust the size and add text
        # colored_attention = Image.fromarray(colored_attention).resize((256, 256))
        # image_with_text = ptp_utils.text_under_image(np.array(colored_attention), decoder(int(tokens[i])))
        # images.append(image_with_text)

    ptp_utils.view_images(np.stack(images, axis=0))
    images = np.stack(images, axis=0)

    num_rows = 1
    offset_ratio = 0.02
    """ Displays a list of images in a grid. """
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

    os.makedirs(save_dir, exist_ok=True)
    pil_img.save(os.path.join(save_dir, f"timestep_{t}_{photo_num}.png"))



def show_image_relevance(image_relevance, image: Image.Image, relevnace_res=16):
    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    image = image.resize((relevnace_res ** 2, relevnace_res ** 2))
    image = np.array(image)

    image_relevance = image_relevance.reshape(1, 1, image_relevance.shape[-1], image_relevance.shape[-1])
    image_relevance = image_relevance.cuda() # because float16 precision interpolation is not supported on cpu
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=relevnace_res ** 2, mode='bilinear')
    image_relevance = image_relevance.cpu() # send it back to cpu
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    image_relevance = image_relevance.reshape(relevnace_res ** 2, relevnace_res ** 2)
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


def get_image_grid(images: List[Image.Image]) -> Image:
    num_images = len(images)
    cols = int(math.ceil(math.sqrt(num_images)))
    rows = int(math.ceil(num_images / cols))
    width, height = images[0].size
    grid_image = Image.new('RGB', (cols * width, rows * height))
    for i, img in enumerate(images):
        x = i % cols
        y = i // cols
        grid_image.paste(img, (x * width, y * height))
    return grid_image


def vis_ann(attention_images, threshold=40):
    att_image = attention_images[:256, :, :]
    att = ((0.2989*att_image[:, :, 0] + 0.5870*att_image[:, :, 1] + 0.1140*att_image[:, :, 2])).astype(int)
    # 0.2989 R + 0.5870 G + 0.1140 B 
    # att = Image.fromarray(att, 'RGB')
    # att.show()
    # att = att.convert('L')
    # att.show()
    # att = np.array(att)

    # print(att.shape)
    # print(att)
    # print(np.min(att), np.max(att))
    plt.imshow(att)
    # for i in range(np.min(att), np.max(att)):
        # print(i, np.sum(att==i))

    att_median = np.max(att) - threshold
    # print("median:", att_median)
    att_show = att.copy()
    att_show[att < att_median] = 255
    att_show[att >= att_median] = 0
    # print(att_show)
    # print("min:", np.min(att_show),"max:", np.max(att_show))
    # img = Image.fromarray(att_show, 'L')
    # img.show()
    plt.imshow(att_show)
    return att_show