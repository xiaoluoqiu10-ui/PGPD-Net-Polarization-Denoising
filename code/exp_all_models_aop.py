"""
一次性跑完所有消融变体 + 推理时间修正
输出 → exp_all_results/full_table.json (论文表格直接用)

模型映射:
  NET1 = 消融1结果9轮/XRnet1.py → 基线 (无辅助模块)
  NET2 = 3_1-9-35/S_Net3_1.py → +动态卷积+偏振注意力
  NET3 = test/3-8-12模型结果 → +噪声估计器 (使用 S_Net3.py)
  NET4 = 3-2训练9轮结果/Snet3_2.py → PGPD-Net 全有

用法:
  python exp_all_models_aop.py
"""

import os, sys, json, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

PROJECT_ROOT = r'D:\LZ\zhoukai_paper'
LZ_DIR = os.path.join(PROJECT_ROOT, 'lz')
sys.path.insert(0, LZ_DIR)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ======================= 路径配置 =======================
JSON_PATH = os.path.join(PROJECT_ROOT, r'2\new\data2.1\dataset_info.json')
OLD_PREFIX = r'G:\zzz\2'
NEW_PREFIX = os.path.join(PROJECT_ROOT, '2')
PATCH_SIZE, STRIDE = 128, 64
OUTPUT_DIR = os.path.join(LZ_DIR, 'exp_all_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = {
    'NET1_baseline': {
        'desc': 'NET1: Baseline (no modules)',
        'weight': os.path.join(LZ_DIR, '消融1结果9轮', 'best_model.pth'),
        'script': os.path.join(LZ_DIR, '消融1结果9轮', 'XRnet1.py'),
        'has_noise_est': False,
    },
    'NET2_dynconv_attn': {
        'desc': 'NET2: +DynamicConv +PolarAttention',
        'weight': os.path.join(LZ_DIR, '3_1-9-35', 'best_model.pth'),
        'script': os.path.join(LZ_DIR, '3_1-9-35', 'S_Net3_1.py'),
        'has_noise_est': False,
    },
    'NET3_noise_est': {
        'desc': 'NET3: +NoiseEstimator',
        'weight': os.path.join(LZ_DIR, 'test', '3-8-12模型结果', 'best_model.pth'),
        'script': os.path.join(LZ_DIR, 'S_Net3.py'),
        'has_noise_est': True,
    },
    'NET4_PGPD_Net': {
        'desc': 'NET4: PGPD-Net (All modules)',
        'weight': os.path.join(LZ_DIR, '3-2训练9轮结果', 'best_model.pth'),
        'script': os.path.join(LZ_DIR, 'Snet3_2.py'),
        'has_noise_est': True,
    },
}


# ======================= 辅助函数 =======================
def remap_path(p):
    if p.startswith(OLD_PREFIX):
        return p.replace(OLD_PREFIX, NEW_PREFIX, 1)
    return p

def read_image(path):
    with Image.open(path) as img:
        return np.array(img.convert('L'), dtype=np.float32) / 255.0

def gaussian_kernel(size, device='cpu'):
    sigma = size / 3.0
    ax = torch.arange(size).float().to(device) - (size - 1.0) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / kernel.max()

def compute_aop(I0, I45, I90, I135):
    S1 = I0 - I90
    S2 = I45 - I135
    return 0.5 * torch.atan2(S2, S1) * 180.0 / np.pi

def compute_psnr(pred, target, max_val=1.0):
    mse = F.mse_loss(pred.squeeze(), target.squeeze())
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(max_val**2 / mse).item()

def compute_ssim(pred, target, data_range=1.0):
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    pred = pred.squeeze().unsqueeze(0).unsqueeze(0)
    target = target.squeeze().unsqueeze(0).unsqueeze(0)
    kernel = torch.ones(1, 1, 11, 11, device=pred.device) / 121.0
    mu1 = F.conv2d(pred, kernel, padding=5)
    mu2 = F.conv2d(target, kernel, padding=5)
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1*mu2
    sigma1_sq = F.conv2d(pred*pred, kernel, padding=5) - mu1_sq
    sigma2_sq = F.conv2d(target*target, kernel, padding=5) - mu2_sq
    sigma12 = F.conv2d(pred*target, kernel, padding=5) - mu1_mu2
    ssim_map = ((2*mu1_mu2+C1)*(2*sigma12+C2))/((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
    return ssim_map.mean().item()

def predict_full_image(model, noisy_tensor, device, has_noise_est=True):
    """滑动窗口 + 高斯融合"""
    model.eval()
    with torch.no_grad():
        C, H, W = noisy_tensor.shape
        accum = {k: torch.zeros((H, W), device=device)
                 for k in ['I0', 'I45', 'I90', 'I135', 'dolp']}
        ws = torch.zeros((H, W), device=device)
        gk = gaussian_kernel(PATCH_SIZE, device=device)

        tops = list(range(0, H-PATCH_SIZE+1, STRIDE))
        if tops[-1] != H-PATCH_SIZE:
            tops.append(H-PATCH_SIZE)
        lefts = list(range(0, W-PATCH_SIZE+1, STRIDE))
        if lefts[-1] != W-PATCH_SIZE:
            lefts.append(W-PATCH_SIZE)

        for top in tops:
            for left in lefts:
                patch = noisy_tensor[:, top:top+PATCH_SIZE, left:left+PATCH_SIZE].unsqueeze(0).to(device)
                try:
                    out = model(patch[:,0:1], patch[:,1:2], patch[:,2:3], patch[:,3:4])
                except TypeError:
                    out = model(patch[:,0:1], patch[:,1:2], patch[:,2:3], patch[:,3:4],
                               noise_level_gt=None)
                for k in accum:
                    accum[k][top:top+PATCH_SIZE, left:left+PATCH_SIZE] += out[k][0,0] * gk
                ws[top:top+PATCH_SIZE, left:left+PATCH_SIZE] += gk

        ws = torch.clamp(ws, min=1e-8)
        return {k: torch.clamp(accum[k]/ws, 0, 1) for k in accum}


# ======================= 加载模型 =======================
def load_model(name, config):
    """动态导入模型类并加载权重"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, config['script'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # 找到 PolarDenoiseNet 类
    model_cls = getattr(module, 'PolarDenoiseNet', None)
    if model_cls is None:
        for attr in dir(module):
            obj = getattr(module, attr)
            if isinstance(obj, type) and 'Polar' in attr:
                model_cls = obj
                break
    if model_cls is None:
        raise RuntimeError(f"Cannot find model class in {config['script']}")

    # 实例化模型
    try:
        model = model_cls(base_channels=32, noise_feat_dim=32).to(DEVICE)
    except TypeError:
        try:
            model = model_cls(base_channels=32).to(DEVICE)
        except TypeError:
            model = model_cls().to(DEVICE)

    # 加载权重
    ckpt = torch.load(config['weight'], map_location=DEVICE)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # 参数量
    params = sum(p.numel() for p in model.parameters()) / 1e6
    return model, params


# ======================= 测量单 patch 推理时间 =======================
def measure_patch_time(model):
    """测量单个 128x128 patch 的推理时间"""
    I0 = torch.randn(1, 1, PATCH_SIZE, PATCH_SIZE).to(DEVICE)
    I45 = torch.randn(1, 1, PATCH_SIZE, PATCH_SIZE).to(DEVICE)
    I90 = torch.randn(1, 1, PATCH_SIZE, PATCH_SIZE).to(DEVICE)
    I135 = torch.randn(1, 1, PATCH_SIZE, PATCH_SIZE).to(DEVICE)

    model.eval()
    # 预热
    with torch.no_grad():
        for _ in range(20):
            try:
                _ = model(I0, I45, I90, I135)
            except TypeError:
                _ = model(I0, I45, I90, I135, noise_level_gt=None)

    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            for _ in range(200):
                try:
                    _ = model(I0, I45, I90, I135)
                except TypeError:
                    _ = model(I0, I45, I90, I135, noise_level_gt=None)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / 200
    else:
        times = []
        with torch.no_grad():
            for _ in range(50):
                t0 = time.perf_counter()
                try:
                    _ = model(I0, I45, I90, I135)
                except TypeError:
                    _ = model(I0, I45, I90, I135, noise_level_gt=None)
                times.append((time.perf_counter()-t0)*1000)
        ms = np.mean(times)
    return ms


# ======================= 主逻辑 =======================
def main():
    # 加载数据索引
    with open(JSON_PATH, 'r') as f:
        all_scenes = json.load(f)

    test_samples = []
    for sd in all_scenes:
        gt = sd['gt_images']
        noise = sd['noise_images']
        for level, paths in noise.items():
            test_samples.append({
                'scene': sd['scene'],
                'noise': level,
                'noise_paths': [remap_path(paths[k]) for k in ['I0','I45','I90','I135']],
                'gt_paths': [remap_path(gt[k]) for k in ['I0','I45','I90','I135','dolp']],
            })

    print(f"Total test samples: {len(test_samples)}")

    all_results = {}

    for model_key, config in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Testing: {config['desc']}")
        print(f"  Weight: {config['weight']}")
        print(f"  Script: {config['script']}")

        try:
            model, params = load_model(model_key, config)
            patch_ms = measure_patch_time(model)
            print(f"  Params: {params:.4f}M, Patch time: {patch_ms:.2f}ms")

            metrics = {k: [] for k in ['I0','I45','I90','I135','dolp','aop_rmse','aop_mae']}
            per_noise = {}

            for idx, s in enumerate(test_samples):
                try:
                    noisy = np.stack([read_image(p) for p in s['noise_paths']], axis=0)
                    noisy_t = torch.from_numpy(noisy).float().to(DEVICE)
                    gt_imgs = [torch.from_numpy(read_image(p)).float().to(DEVICE)
                               for p in s['gt_paths']]
                    gt_I0, gt_I45, gt_I90, gt_I135, gt_dolp = gt_imgs
                    gt_aop = compute_aop(gt_I0, gt_I45, gt_I90, gt_I135)

                    pred = predict_full_image(model, noisy_t, DEVICE, config['has_noise_est'])
                    pI0, pI45, pI90, pI135, pDolp = pred['I0'], pred['I45'], pred['I90'], pred['I135'], pred['dolp']
                    pAop = compute_aop(pI0, pI45, pI90, pI135)

                    for key, pimg, gimg in [
                        ('I0', pI0, gt_I0), ('I45', pI45, gt_I45),
                        ('I90', pI90, gt_I90), ('I135', pI135, gt_I135),
                        ('dolp', pDolp, gt_dolp),
                    ]:
                        metrics[key].append({
                            'psnr': compute_psnr(pimg, gimg),
                            'ssim': compute_ssim(pimg, gimg),
                        })

                    aop_diff = torch.abs(pAop - gt_aop)
                    metrics['aop_rmse'].append(torch.sqrt(torch.mean(aop_diff**2)).item())
                    metrics['aop_mae'].append(torch.mean(aop_diff).item())

                    # 按噪声分组
                    nl = s['noise']
                    if nl not in per_noise:
                        per_noise[nl] = {k: [] for k in ['dolp_psnr','dolp_ssim','aop_rmse']}
                    per_noise[nl]['dolp_psnr'].append(metrics['dolp'][-1]['psnr'])
                    per_noise[nl]['dolp_ssim'].append(metrics['dolp'][-1]['ssim'])
                    per_noise[nl]['aop_rmse'].append(metrics['aop_rmse'][-1])

                except Exception as e:
                    pass

                if (idx+1) % 20 == 0:
                    print(f"  Progress: {idx+1}/{len(test_samples)}")

            # 汇总
            result = {
                'desc': config['desc'],
                'params_M': round(params, 4),
                'patch_time_ms': round(patch_ms, 2),
            }
            for key in ['I0','I45','I90','I135','dolp']:
                result[f'{key}_psnr'] = round(np.mean([m['psnr'] for m in metrics[key]]), 2)
                result[f'{key}_ssim'] = round(np.mean([m['ssim'] for m in metrics[key]]), 4)
            result['aop_rmse_deg'] = round(np.mean(metrics['aop_rmse']), 2)
            result['aop_mae_deg'] = round(np.mean(metrics['aop_mae']), 2)

            # 按噪声
            result['by_noise'] = {}
            for nl in sorted(per_noise.keys()):
                result['by_noise'][nl] = {
                    'dolp_psnr': round(np.mean(per_noise[nl]['dolp_psnr']), 2),
                    'dolp_ssim': round(np.mean(per_noise[nl]['dolp_ssim']), 4),
                    'aop_rmse': round(np.mean(per_noise[nl]['aop_rmse']), 2),
                }

            all_results[model_key] = result

            # 打印
            print(f"\n  --- {config['desc']} ---")
            print(f"  I0={result['I0_psnr']:.2f}/{result['I0_ssim']:.4f}  "
                  f"I45={result['I45_psnr']:.2f}/{result['I45_ssim']:.4f}  "
                  f"I90={result['I90_psnr']:.2f}/{result['I90_ssim']:.4f}  "
                  f"I135={result['I135_psnr']:.2f}/{result['I135_ssim']:.4f}")
            print(f"  DoLP={result['dolp_psnr']:.2f}/{result['dolp_ssim']:.4f}  "
                  f"AoP RMSE={result['aop_rmse_deg']:.2f}deg  "
                  f"Params={result['params_M']:.2f}M  "
                  f"Time={result['patch_time_ms']:.2f}ms/patch")

        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_results[model_key] = {'error': str(e)}

    # ======================= 保存 =======================
    out_path = os.path.join(OUTPUT_DIR, 'full_table.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # ======================= 打印完整论文表格 =======================
    print("\n\n" + "=" * 90)
    print("                    COMPLETE ABLATION TABLE (Paper-Ready)")
    print("=" * 90)

    header = "{:<20} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}".format(
        'Model', 'I0', 'I45', 'I90', 'I135', 'DoLP', 'AoP(RMSE)', 'Params(M)', 'Time(ms)')
    print(header)
    print("-" * 90)

    for mk in ['NET1_baseline', 'NET2_dynconv_attn', 'NET3_noise_est', 'NET4_PGPD_Net']:
        if mk not in all_results or 'error' in all_results[mk]:
            print(f"{mk:<20} {'FAILED':>8}")
            continue
        r = all_results[mk]
        name = {'NET1_baseline': 'NET1 (Baseline)',
                'NET2_dynconv_attn': 'NET2 (+DynConv+Attn)',
                'NET3_noise_est': 'NET3 (+NoiseEst)',
                'NET4_PGPD_Net': 'NET4 (PGPD-Net)'}.get(mk, mk)
        print("{:<20} {:>8.2f} {:>8.2f} {:>8.2f} {:>8.2f} {:>8.2f} {:>8.2f} {:>8.2f} {:>8.2f}".format(
            name, r['I0_psnr'], r['I45_psnr'], r['I90_psnr'], r['I135_psnr'],
            r['dolp_psnr'], r['aop_rmse_deg'], r['params_M'], r['patch_time_ms']))

    print("-" * 90)
    print(f"\nResult saved to: {out_path}")
    print("Done.")


if __name__ == '__main__':
    main()
