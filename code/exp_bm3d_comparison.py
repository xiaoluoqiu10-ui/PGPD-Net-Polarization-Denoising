"""
BM3D 对比实验 — 在你的测试集上运行 BM3D 去噪
计算全部指标: I0/I45/I90/I135/DoLP/AoP 的 PSNR + SSIM

用法:
  python exp_bm3d_comparison.py
"""

import os, json, time, sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = r'D:\LZ\zhoukai_paper'
LZ_DIR = os.path.join(PROJECT_ROOT, 'lz')
JSON_PATH = os.path.join(PROJECT_ROOT, r'2\new\data2.1\dataset_info.json')
OLD_PREFIX = r'G:\zzz\2'
NEW_PREFIX = os.path.join(PROJECT_ROOT, '2')
OUTPUT_DIR = os.path.join(LZ_DIR, 'exp_bm3d_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def remap_path(p):
    if p.startswith(OLD_PREFIX):
        return p.replace(OLD_PREFIX, NEW_PREFIX, 1)
    return p


def read_image(path):
    with Image.open(path) as img:
        return np.array(img.convert('L'), dtype=np.float32) / 255.0


def compute_aop(I0, I45, I90, I135):
    S1 = I0 - I90
    S2 = I45 - I135
    return 0.5 * np.arctan2(S2, S1) * 180.0 / np.pi


def compute_dolp(I0, I45, I90, I135):
    S0 = (I0 + I45 + I90 + I135) / 2.0
    S0 = np.clip(S0, 1e-6, None)
    S1 = I0 - I90
    S2 = I45 - I135
    dolp = np.sqrt(S1**2 + S2**2) / S0
    return np.clip(dolp, 0, 1)


def compute_psnr_torch(pred, target, max_val=1.0):
    p = torch.from_numpy(pred).float()
    t = torch.from_numpy(target).float()
    mse = F.mse_loss(p, t)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(max_val**2 / mse).item()


def compute_ssim_torch(pred, target):
    p = torch.from_numpy(pred).float().unsqueeze(0).unsqueeze(0)
    t = torch.from_numpy(target).float().unsqueeze(0).unsqueeze(0)
    C1, C2 = 0.01**2, 0.03**2
    kernel = torch.ones(1, 1, 11, 11) / 121.0
    mu1 = F.conv2d(p, kernel, padding=5)
    mu2 = F.conv2d(t, kernel, padding=5)
    s1 = F.conv2d(p*p, kernel, padding=5) - mu1**2
    s2 = F.conv2d(t*t, kernel, padding=5) - mu2**2
    s12 = F.conv2d(p*t, kernel, padding=5) - mu1*mu2
    ssim = ((2*mu1*mu2+C1)*(2*s12+C2))/((mu1**2+mu2**2+C1)*(s1+s2+C2))
    return ssim.mean().item()


def bm3d_denoise_angle(noisy_img, sigma=25):
    """用 BM3D 对单通道图像去噪"""
    import bm3d
    # BM3D expects [0,255] uint8
    img_uint8 = (np.clip(noisy_img, 0, 1) * 255).astype(np.uint8)
    denoised = bm3d.bm3d(img_uint8, sigma_psd=sigma, stage_arg=bm3d.BM3DStages.ALL_STAGES)
    return denoised.astype(np.float32) / 255.0


def main():
    print("Loading test data index...")
    with open(JSON_PATH, 'r') as f:
        all_scenes = json.load(f)

    test_samples = []
    for sd in all_scenes:
        gt = sd['gt_images']
        noise = sd['noise_images']
        for level, paths in noise.items():
            test_samples.append({
                'scene': sd['scene'], 'noise': level,
                'noise_paths': [remap_path(paths[k]) for k in ['I0','I45','I90','I135']],
                'gt_paths': [remap_path(gt[k]) for k in ['I0','I45','I90','I135','dolp']],
            })

    print(f"Test samples: {len(test_samples)}")

    # 噪声水平 → BM3D sigma 映射
    # BM3D sigma 范围 0-255, 对应噪声强度
    noise_sigma_map = {'500': 35, '1000': 25, '2000': 18, '4000': 12}

    all_metrics = []
    t0 = time.time()

    for idx, s in enumerate(test_samples):
        try:
            noisy_imgs = [read_image(p) for p in s['noise_paths']]
            gt_imgs = [read_image(p) for p in s['gt_paths']]

            # 选择 sigma
            sigma = noise_sigma_map.get(s['noise'], 25)

            # BM3D 逐角度去噪
            denoised = []
            for i, angle_name in enumerate(['I0', 'I45', 'I90', 'I135']):
                img_den = bm3d_denoise_angle(noisy_imgs[i], sigma=sigma)
                denoised.append(img_den)

            # 计算 DoLP 和 AoP
            dolp_pred = compute_dolp(*denoised)
            aop_pred = compute_aop(*denoised)
            aop_gt = compute_aop(*gt_imgs[:4])
            dolp_gt = gt_imgs[4]

            row = {'scene': s['scene'], 'noise': s['noise']}
            for i, name in enumerate(['I0','I45','I90','I135']):
                row[f'{name}_psnr'] = round(compute_psnr_torch(denoised[i], gt_imgs[i]), 2)
                row[f'{name}_ssim'] = round(compute_ssim_torch(denoised[i], gt_imgs[i]), 4)
            row['dolp_psnr'] = round(compute_psnr_torch(dolp_pred, dolp_gt), 2)
            row['dolp_ssim'] = round(compute_ssim_torch(dolp_pred, dolp_gt), 4)
            row['aop_rmse'] = round(float(np.sqrt(np.mean((aop_pred - aop_gt)**2))), 2)
            all_metrics.append(row)

        except Exception as e:
            print(f"  Sample {idx} failed: {e}")

        if (idx+1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (idx+1) * (len(test_samples)-idx-1)
            print(f"  [{idx+1}/{len(test_samples)}] elapsed={elapsed:.0f}s, eta={eta:.0f}s")

    # 汇总
    avg = {}
    keys = ['I0_psnr','I45_psnr','I90_psnr','I135_psnr',
            'I0_ssim','I45_ssim','I90_ssim','I135_ssim',
            'dolp_psnr','dolp_ssim','aop_rmse']
    for k in keys:
        avg[k] = round(np.mean([m[k] for m in all_metrics]), 4 if 'ssim' in k else 2)

    print("\n" + "="*70)
    print("               BM3D Denoising Results on Our Dataset")
    print("="*70)
    print(f"I0:   PSNR={avg['I0_psnr']:.2f}  SSIM={avg['I0_ssim']:.4f}")
    print(f"I45:  PSNR={avg['I45_psnr']:.2f}  SSIM={avg['I45_ssim']:.4f}")
    print(f"I90:  PSNR={avg['I90_psnr']:.2f}  SSIM={avg['I90_ssim']:.4f}")
    print(f"I135: PSNR={avg['I135_psnr']:.2f}  SSIM={avg['I135_ssim']:.4f}")
    print(f"DoLP: PSNR={avg['dolp_psnr']:.2f}  SSIM={avg['dolp_ssim']:.4f}")
    print(f"AoP:  RMSE={avg['aop_rmse']:.2f}deg")
    print("="*70)

    # 保存
    result = {'BM3D_average': avg, 'per_sample': all_metrics}
    out_path = os.path.join(OUTPUT_DIR, 'bm3d_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {out_path}")
    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
