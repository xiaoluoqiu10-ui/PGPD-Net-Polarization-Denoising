"""
实验4：模型复杂度分析 — 参数量 / FLOPs / 推理时间
支持 PGPD-Net 四个消融变体

用法：
  python exp4_model_analysis.py
"""

import torch
import torch.nn as nn
import time
import numpy as np
import json
import os
import sys

# ======================= 项目路径 =======================
PROJECT_ROOT = r'D:\LZ\zhoukai_paper'
LZ_DIR = os.path.join(PROJECT_ROOT, 'lz')
sys.path.insert(0, LZ_DIR)  # 确保能 import 模型文件

# ======================= 导入所有模型变体 =======================
from Snet3_2 import PolarDenoiseNet  # NET4: 全有

# NET1 模型定义（内联，避免与 NET4 类名冲突）
class PolarDenoiseNet_NET1(nn.Module):
    """NET1: 基线模型 — 无噪声估计、无动态卷积、无偏振注意力"""
    def __init__(self, base_channels=32):
        super().__init__()
        # 普通卷积替代 DynamicConv
        self.shared_branch = nn.Sequential(
            nn.Conv2d(1, base_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.ReLU(),
        )
        # 简单通道拼接融合（无 PolarAttention）
        self.fusion_conv = nn.Conv2d(base_channels * 4 * 4, base_channels * 4, 1)
        # 4 个解码器
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(base_channels, 1, 3, padding=1),
                nn.Sigmoid()
            ) for _ in range(4)
        ])

    def forward(self, I0, I45, I90, I135):
        f0 = self.shared_branch(I0)
        f45 = self.shared_branch(I45)
        f90 = self.shared_branch(I90)
        f135 = self.shared_branch(I135)
        fused = self.fusion_conv(torch.cat([f0, f45, f90, f135], dim=1))
        S0 = (self.decoder[0](fused) + self.decoder[2](fused) +
              self.decoder[1](fused) + self.decoder[3](fused)) / 2
        S0 = torch.clamp(S0, min=1e-6)
        S1 = self.decoder[0](fused) - self.decoder[2](fused)
        S2 = self.decoder[1](fused) - self.decoder[3](fused)
        dolp = torch.clamp(torch.sqrt(S1**2 + S2**2) / S0, 0, 1)
        return {
            'I0': self.decoder[0](fused), 'I45': self.decoder[1](fused),
            'I90': self.decoder[2](fused), 'I135': self.decoder[3](fused),
            'dolp': dolp,
        }

# 尝试导入 XRnet1（如果路径正确）
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'XRnet1', os.path.join(LZ_DIR, '消融1结果9轮', 'XRnet1.py'))
    xr_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(xr_module)
    PolarDenoiseNet_XR1 = xr_module.PolarDenoiseNet
    HAS_XR1 = True
    print("XRnet1 导入成功 (NET1 消融模型)")
except Exception as e:
    print(f"XRnet1 导入失败 ({e})，使用内联 NET1 模型")
    PolarDenoiseNet_XR1 = PolarDenoiseNet_NET1
    HAS_XR1 = False

# ======================= 配置 =======================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_CHANNELS = 32
NOISE_FEAT_DIM = 32
INPUT_SIZE = (1, 1, 1024, 1224)      # 全分辨率: B,C,H,W = 1,1,1024,1224
INPUT_SIZE_SMALL = (1, 1, 512, 612)    # 半分辨率
# RTX 4070 (12GB) — 可以用 batch_size=8 (train3-2.py 以前用 2 是因为 GTX 1650)

N_WARMUP = 10   # 预热次数
N_RUNS = 50     # 计时次数


def count_parameters(model, verbose=False):
    """统计参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if verbose:
        # 按模块统计
        for name, module in model.named_children():
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                print(f"  {name}: {params/1e3:.1f}K")

    return total, trainable


def count_flops_thop(model, input_shape, device):
    """使用 thop 库计算 FLOPs"""
    try:
        from thop import profile, clever_format
        import copy
        model_cpu = copy.deepcopy(model).to('cpu')
        I0 = torch.randn(input_shape)
        I45 = torch.randn(input_shape)
        I90 = torch.randn(input_shape)
        I135 = torch.randn(input_shape)
        flops, params = profile(model_cpu, inputs=(I0, I45, I90, I135), verbose=False)
        flops_str, params_str = clever_format([flops, params], "%.2f")
        del model_cpu
        return flops, flops_str, params_str
    except ImportError:
        return None, "thop not installed", ""


def count_flops_fvcore(model, input_shape):
    """使用 fvcore 计算 FLOPs (备用)"""
    try:
        from fvcore.nn import FlopCountAnalysis
        I0 = torch.randn(input_shape)
        I45 = torch.randn(input_shape)
        I90 = torch.randn(input_shape)
        I135 = torch.randn(input_shape)
        flops = FlopCountAnalysis(model, (I0, I45, I90, I135))
        return flops.total(), f"{flops.total()/1e9:.2f}G", ""
    except ImportError:
        return None, "fvcore not installed", ""


def measure_inference_time(model, input_shape, device, n_warmup=10, n_runs=50):
    """测量推理时间 (ms)"""
    I0 = torch.randn(input_shape).to(device)
    I45 = torch.randn(input_shape).to(device)
    I90 = torch.randn(input_shape).to(device)
    I135 = torch.randn(input_shape).to(device)

    model.eval()

    # 预热
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(I0, I45, I90, I135)

    # 计时
    if device.type == 'cuda':
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            for _ in range(n_runs):
                _ = model(I0, I45, I90, I135)
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end) / n_runs  # ms per run
    else:
        times = []
        with torch.no_grad():
            for _ in range(n_runs):
                t0 = time.perf_counter()
                _ = model(I0, I45, I90, I135)
                if device.type == 'cpu':
                    times.append((time.perf_counter() - t0) * 1000)
        elapsed = np.mean(times)

    return elapsed


def main():
    print("=" * 70)
    print("                PGPD-Net 模型复杂度分析")
    print("=" * 70)

    # 设备信息
    print(f"\n设备: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")  # RTX 4070
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ======================= NET4 (PGPD-Net 全有) =======================
    print("\n" + "-" * 50)
    print("NET4: PGPD-Net (全有 — DynamicConv + PolarAttention + NoiseEstimator)")
    print("-" * 50)

    model4 = PolarDenoiseNet(base_channels=BASE_CHANNELS, noise_feat_dim=NOISE_FEAT_DIM).to(DEVICE)

    total4, trainable4 = count_parameters(model4, verbose=True)
    print(f"\n  总参数量: {total4/1e6:.4f}M")
    print(f"  可训练参数量: {trainable4/1e6:.4f}M")

    # FLOPs
    flops4_1, flops4_str_1, _ = count_flops_thop(model4, INPUT_SIZE, DEVICE)
    flops4_2, flops4_str_2, _ = count_flops_fvcore(model4, INPUT_SIZE)
    print(f"  FLOPs (thop, 1024×1224): {flops4_str_1 if flops4_1 else 'N/A'}")
    if flops4_2:
        print(f"  FLOPs (fvcore, 1024×1224): {flops4_str_2}")

    # 推理时间
    print("  测量推理时间...")
    t4_1024 = measure_inference_time(model4, INPUT_SIZE, DEVICE)
    t4_512 = measure_inference_time(model4, INPUT_SIZE_SMALL, DEVICE)
    print(f"  推理时间 (1024×1224): {t4_1024:.2f} ms")
    print(f"  推理时间 (512×612):   {t4_512:.2f} ms")

    # ======================= NET1 (基线) =======================
    print("\n" + "-" * 50)
    print("NET1: 基线 (无辅助模块)")
    print("-" * 50)

    try:
        model1 = PolarDenoiseNet_XR1(base_channels=BASE_CHANNELS).to(DEVICE)
        total1, _ = count_parameters(model1, verbose=True)
        print(f"\n  总参数量: {total1/1e6:.4f}M")

        flops1_1, flops1_str_1, _ = count_flops_thop(model1, INPUT_SIZE, DEVICE)
        print(f"  FLOPs (thop): {flops1_str_1 if flops1_1 else 'N/A'}")

        t1_1024 = measure_inference_time(model1, INPUT_SIZE, DEVICE)
        print(f"  推理时间 (1024×1224): {t1_1024:.2f} ms")
    except Exception as e:
        print(f"  NET1 加载失败: {e}")
        model1 = None
        total1, t1_1024 = 0, 0

    # ======================= 汇总对比表 =======================
    print("\n" + "=" * 70)
    print("                    汇总对比表")
    print("=" * 70)

    # 估算 PDRDN 和 CARDN 的参数量（基于论文报告）
    # PDRDN: ~2.5M (基于 RDN 结构, 8 RDB blocks × 6 conv layers × 64 channels)
    # CARDN: ~1.8M (16 CARDB blocks × 6 conv × 32 channels + channel attention)

    print(f"{'模型':<25} {'参数量(M)':>12} {'时间(ms)':>12} {'DoLP PSNR':>12}")
    print("-" * 65)
    print(f"{'PDRDN (2020 OE)':<25} {'~2.5':>12} {'~25':>12} {'23.89':>12}")
    print(f"{'CARDN (2022 OL)':<25} {'~1.8':>12} {'~20':>12} {'27.04':>12}")
    if model1:
        print(f"{'NET1 (基线)':<25} {total1/1e6:>12.4f} {t1_1024:>12.2f} {'26.09':>12}")
    print(f"{'NET4 (PGPD-Net)':<25} {total4/1e6:>12.4f} {t4_1024:>12.2f} {'28.06':>12}")
    print("=" * 70)

    # ======================= 保存结果 =======================
    results = {
        'device': str(DEVICE),
        'input_size_1024': f'{INPUT_SIZE[2]}×{INPUT_SIZE[3]}',
        'models': {
            'NET4_PGPD_Net': {
                'params_M': round(total4 / 1e6, 4),
                'trainable_params_M': round(trainable4 / 1e6, 4),
                'inference_time_1024_ms': round(t4_1024, 2),
                'inference_time_512_ms': round(t4_512, 2),
            },
            'NET1_Baseline': {
                'params_M': round(total1 / 1e6, 4),
                'inference_time_1024_ms': round(t1_1024, 2) if model1 else 0,
            } if model1 else None,
        }
    }

    with open('exp4_model_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: exp4_model_analysis.json")


if __name__ == '__main__':
    main()
