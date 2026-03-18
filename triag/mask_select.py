import os, glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

def ssim_gray(x: np.ndarray, y: np.ndarray, win_size: int = 11, sigma: float = 1.5) -> float:
    assert x.shape == y.shape, f"SSIM需要同尺寸，got {x.shape} vs {y.shape}"
    x = x.astype(np.float32) / 255.0
    y = y.astype(np.float32) / 255.0
    k = win_size if win_size % 2 == 1 else win_size + 1
    mu1 = cv2.GaussianBlur(x, (k, k), sigma)
    mu2 = cv2.GaussianBlur(y, (k, k), sigma)
    mu1_sq, mu2_sq, mu12 = mu1*mu1, mu2*mu2, mu1*mu2
    sigma1_sq = cv2.GaussianBlur(x*x, (k, k), sigma) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(y*y, (k, k), sigma) - mu2_sq
    sigma12   = cv2.GaussianBlur(x*y, (k, k), sigma) - mu12
    C1, C2 = (0.01)**2, (0.03)**2
    ssim_map = ((2*mu12 + C1) * (2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-12)
    return float(np.clip(ssim_map.mean(), 0.0, 1.0))

def read_gray(p):
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(p)
    return img

def list_paths(d, recursive=True):
    exts = ["*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.PNG","*.JPG","*.JPEG","*.BMP","*.TIF"]
    out = []
    for e in exts:
        pat = os.path.join(d, "**", e) if recursive else os.path.join(d, e)
        out += glob.glob(pat, recursive=recursive)
    return sorted(out)

def show_best_pairs_two_cols_ssim_pure_sizefilter(
    gen_dir: str, mvtec_dir: str, K: int = 10, recursive=True,
    size_ratio_min: float = 0.65,   # 面积相近筛子：min(area)/max(area) 必须 >= 该值
    binarize_with_otsu: bool = True # 用来数白像素，和 SSIM 无关
):
    gen_paths = list_paths(gen_dir, recursive=recursive)
    mv_paths  = list_paths(mvtec_dir, recursive=recursive)
    print(f"[info] gen={len(gen_paths)}  mvtec={len(mv_paths)}")
    if len(gen_paths) == 0 or len(mv_paths) == 0:
        print("[warn] 目录下没找到图片"); return

    rows = min(max(1, K), len(gen_paths))
    fig, axes = plt.subplots(rows, 2, figsize=(8, 3*rows))
    if rows == 1: axes = np.array([axes])

    mv_imgs = [read_gray(p) for p in mv_paths]

    def white_area(img):
        if binarize_with_otsu:
            _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        else:
            bw = img
        return int((bw > 0).sum())

    results = []
    for i, gp in enumerate(gen_paths[:rows]):
        g = read_gray(gp)

        best_j, best_ssim = -1, -1.0
        best_g_resized = None

        # 先算一次 gen 的白像素面积（注意：需在“与候选同尺寸”下比较）
        for j, m in enumerate(mv_imgs):
            # 把 gen resize 到当前 mvtec 的尺寸
            g_res = g if g.shape == m.shape else cv2.resize(g, (m.shape[1], m.shape[0]), interpolation=cv2.INTER_NEAREST)

            # —— 尺寸护栏：比较白像素面积是否相近（在同一尺寸下比较）——
            ga = white_area(g_res)
            ma = white_area(m)
            ratio = (min(ga, ma) / max(ga, ma)) if max(ga, ma) > 0 else 0.0
            if ratio < size_ratio_min:
                continue  # 尺寸差过大，跳过这一对

            # —— 纯 SSIM ——（不做ROI、不做对齐）
            s = ssim_gray(g_res, m)
            if s > best_ssim:
                best_ssim = s
                best_j = j
                best_g_resized = g_res

        # 如果没有任何候选通过尺寸筛子，就退化为“纯 SSIM 的最佳”
        if best_j == -1:
            fallback_ssim = -1.0
            fb_j = -1
            fb_g_res = None
            for j, m in enumerate(mv_imgs):
                g_res = g if g.shape == m.shape else cv2.resize(g, (m.shape[1], m.shape[0]), interpolation=cv2.INTER_NEAREST)
                s = ssim_gray(g_res, m)
                if s > fallback_ssim:
                    fallback_ssim, fb_j, fb_g_res = s, j, g_res
            best_j, best_ssim, best_g_resized = fb_j, fallback_ssim, fb_g_res

        results.append((gp, mv_paths[best_j], best_ssim))

        # —— 两列展示 —— 
        axL, axR = axes[i, 0], axes[i, 1]
        axL.imshow(best_g_resized, cmap="gray")
        axL.set_title(f"gen(resized): {os.path.basename(gp)}")
        axL.axis("off")
        axR.imshow(mv_imgs[best_j], cmap="gray")
        axR.set_title(f"mvtec: {os.path.basename(mv_paths[best_j])} | SSIM={best_ssim:.4f}")
        axR.axis("off")

    plt.tight_layout(); plt.show()
    return results

def best_mvtec_mask_for_gen(
    gen_mask_path: str,
    mvtec_dir: str,
    recursive: bool = True,
    size_ratio_min: float = 0.65,    # 设 0 关闭尺寸筛子
    binarize_with_otsu: bool = True,
    return_basename: bool = True,     # True 只返回文件名；False 返回完整路径
    class_name: str = None          # 可选，打印时用
):
    mv_paths = list_paths(mvtec_dir, recursive=recursive)
    if len(mv_paths) == 0:
        raise FileNotFoundError(f"No images found in {mvtec_dir}")
    
    if class_name is None:
        first_third = max(1, math.ceil(len(mv_paths) / 3))
        sub_paths = mv_paths[:first_third]
        mv_paths = sub_paths
        # import pdb;pdb.set_trace()
    elif class_name == 'hazelnut':
        sub_paths = [mv_paths[i] for i in (2,3,5,12,13,16)]
        mv_paths = sub_paths
        # import pdb;pdb.set_trace()
    else:
        sub_paths = mv_paths
        mv_paths = sub_paths

    g = read_gray(gen_mask_path)
    mv_imgs = [read_gray(p) for p in mv_paths]

    def white_area(img):
        if binarize_with_otsu:
            _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        else:
            bw = img
        return int((bw > 0).sum())

    best_j, best_ssim = None, -1.0

    # 先在“尺寸相近”的候选里找 SSIM 最大的
    for j, m in enumerate(mv_imgs):
        g_res = g if g.shape == m.shape else cv2.resize(g, (m.shape[1], m.shape[0]), interpolation=cv2.INTER_NEAREST)

        if size_ratio_min > 0:
            ga = white_area(g_res)
            ma = white_area(m)
            ratio = (min(ga, ma) / max(ga, ma)) if max(ga, ma) > 0 else 0.0
            if ratio < size_ratio_min:
                continue

        s = ssim_gray(g_res, m)
        if s > best_ssim:
            best_ssim = s
            best_j = j

    # 若尺寸筛子过滤掉全部候选，则退化为“纯 SSIM 最大”的那张
    if best_j is None:
        for j, m in enumerate(mv_imgs):
            g_res = g if g.shape == m.shape else cv2.resize(g, (m.shape[1], m.shape[0]), interpolation=cv2.INTER_NEAREST)
            s = ssim_gray(g_res, m)
            if s > best_ssim:
                best_ssim = s
                best_j = j

    best_path = mv_paths[best_j]
    return (os.path.basename(best_path) if return_basename else best_path), best_ssim
