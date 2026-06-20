# Code Inventory

## Model Definitions

| File | Model | Description | Key Components |
|------|-------|-------------|---------------|
| `lz/Snet3_2.py` | NET4 (PGPD-Net Full) | All modules | DynamicConv2d + PolarAttention + NoiseLevelEstimator + 4 Decoders |
| `lz/3_1-9-35/S_Net3_1.py` | NET2 (+DynConv+Attn) | **BEST MODEL** | DynamicConv2d + PolarAttention + 4 Decoders (no noise estimator) |
| `lz/S_Net3.py` | NET3 (+NoiseEst) | Noise only | NoiseLevelEstimator + 4 Decoders (no attention) |
| `lz/消融1结果9轮/XRnet1.py` | NET1 (Baseline) | No modules | Basic conv + 4 Decoders |

### NET2 Architecture (Recommended Final Model)
```
Input: I0, I45, I90, I135 (4×[B,1,H,W])
    │
    ├── Shared Feature Extractor (PolarFeatureExtractor)
    │   ├── DynamicConv2d(1→32) — multi-scale gating (1/2/4 pool) + 4 expert convs
    │   ├── Conv(32→64→128)
    │   └── 3× ResBlock(128)
    │   Output: 4× [B,128,H,W]
    │
    ├── PolarAttention
    │   ├── Concat 4 angles → [B,512,H,W]
    │   ├── 1×1 Conv → Sigmoid → 4 attention maps
    │   └── Element-wise weight on each angle feature
    │
    ├── Fusion: 1×1 Conv(512→256→128) → [B,128,H,W]
    │
    ├── 4× Decoder (independent, same structure)
    │   └── 128→64→ResBlock→32→ResBlock→1→Sigmoid
    │
    ├── Physical DoLP: S0=(I0+I45+I90+I135)/2, S1=I0-I90, S2=I45-I135
    │                  DoLP = √(S1²+S2²) / S0
    │
    Output: {I0', I45', I90', I135', DoLP'}
```

### Loss Function
```python
L_total = L1_angle + alpha * L1_dolp
# alpha = 0.5 (fixed)
# gamma = 0.0 (noise estimation loss DISABLED)
```

## Training

| File | Purpose |
|------|---------|
| `lz/train3-2.py` | NET4 训练脚本 (Config: lr=5e-5, batch=2, epochs=35, patch=128, stride=64) |
| `lz/data_loader.py` | 数据集加载 |

## Experiment Scripts

| Script | What It Does | Output |
|------|------|------|
| `exp_all_models_aop.py` | 跑 NET1-4 全部 4 个模型，计算所有指标 | `exp_all_results/full_table.json` |
| `exp_bm3d_comparison.py` | 对 84 测试样本跑 BM3D 去噪+计算指标 | `exp_bm3d_results/bm3d_results.json` |
| `exp_material_analysis.py` | 按材质分组统计 NET2 的 DoLP/AoP | `exp_all_results/material_results.json` |
| `exp_material_mapping.py` | 建立场景→材质映射表 | `exp_all_results/material_mapping.json` |
| `exp_deep_analysis.py` | NET2 vs NET4 深入对比分析 | Console output |
| `exp4_model_analysis.py` | 参数量/FLOPs/推理时间 | `exp4_model_analysis.json` |
| `exp1_compute_aop.py` | 单模型 AoP 评估（已被 exp_all_models_aop.py 取代） | `exp1_aop_results/` |

## Model Weights

| Model | Path | Size |
|-------|------|------|
| NET1 (Baseline) | `lz/消融1结果9轮/best_model.pth` | ~26MB |
| NET2 (+DynConv+Attn) **BEST** | `lz/3_1-9-35/best_model.pth` | ~31MB |
| NET3 (+NoiseEst) | `lz/test/3-8-12模型结果/best_model.pth` | ~26MB |
| NET4 (All modules) | `lz/3-2训练9轮结果/best_model.pth` | ~31MB |
| NET4 (older checkpoint) | `lz/best_model.pth` | ~31MB |

## Key Path Configuration
All scripts need this mapping (old computer → current computer):
```python
OLD_PATH_PREFIX = r'G:\zzz\2'
NEW_PATH_PREFIX = r'D:\LZ\zhoukai_paper\2'
```

## Running Experiments
```bash
cd D:\LZ\zhoukai_paper\lz

# Full ablation (takes ~20 min, needs GPU)
python exp_all_models_aop.py

# BM3D comparison (takes ~40 min, needs GPU)
python exp_bm3d_comparison.py

# Material analysis (takes ~5 min, needs GPU)
python exp_material_analysis.py

# Model efficiency (takes ~1 min, CPU ok)
python exp4_model_analysis.py
```
