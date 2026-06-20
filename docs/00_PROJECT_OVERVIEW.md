# PGPD-Net Project Overview

## 项目目标
将 PGPD-Net (Physics-Guided Polarization Denoising Network) 论文投稿至 **OLT (Optics & Laser Technology)** 或 **OLEN (Optics and Lasers in Engineering)**，两者均为 Elsevier 期刊，共用 `elsarticle.cls` 模板。

## 项目目录结构

```
D:\LZ\zhoukai_paper\
├── 2\                          ← 主数据目录
│   ├── new\data2.1\            ← 训练/测试数据 (21场景 × 4噪声水平)
│   │   ├── 1\ ~ 21\            ← 场景子目录，含 I0/I45/I90/I135 的 noise 和 gt 图像
│   │   ├── dataset_info.json   ← 数据索引文件
│   │   └── comparedata\        ← 对比方法数据
│   └── data\                   ← 旧版数据
├── lz\                         ← 代码目录 (PGPD-Net)
│   ├── Snet3_2.py              ← NET4 模型定义 (PGPD-Net 全有)
│   ├── S_Net3_1.py             ← NET2 模型定义 (+DynConv+Attn)
│   ├── S_Net3.py               ← NET3 模型定义 (+NoiseEst)
│   ├── train3-2.py             ← 训练脚本
│   ├── 3-2训练9轮结果\          ← NET4 训练结果 (best model)
│   ├── 3_1-9-35\               ← NET2 训练结果 (best model)
│   ├── 消融1结果9轮\            ← NET1 训练结果 (best model)
│   ├── test\3-8-12模型结果\     ← NET3 训练结果 (best model)
│   ├── exp1_compute_aop.py     ← AoP 评估脚本
│   ├── exp4_model_analysis.py  ← 模型复杂度分析
│   ├── exp_all_models_aop.py   ← 4模型完整消融评估
│   ├── exp_bm3d_comparison.py  ← BM3D 对比实验
│   ├── exp_material_analysis.py← 多材质分析
│   ├── exp_deep_analysis.py    ← NET2 vs NET4 深度分析
│   ├── exp_all_results\        ← 实验结果输出
│   └── exp_bm3d_results\       ← BM3D 结果输出
├── papers_downloaded\          ← 下载的参考文献
│   ├── 07_CARDN_*.pdf          ← CARDN (OL 2022)
│   ├── 05_Review_*.pdf         ← Review (Adv Imaging 2024)
│   ├── 02_Polarized_Color_*.pdf← CVPR 2023
│   └── 01_PDD_*.pdf            ← PDD Dataset (ICIP 2025)
├── datasets\PDD\               ← PDD公开数据集 (下载中断, 735MB/12.2GB)
├── olt_paper\                  ← LaTeX 模板目录
│   └── elsarticle\             ← Elsevier elsarticle.cls 模板
├── project_knowledge\          ← 本知识转移目录
└── 第二部分初稿.docx            ← 论文中文初稿
```

## GPU 环境
- **GPU**: NVIDIA GeForce RTX 4070 (12GB VRAM)
- **Python**: 3.10 (miniconda3)
- **PyTorch**: CUDA 版本
- **关键包**: torch, numpy, PIL, pdfplumber, pypdf, bm3d, polanalyser, thop

## 关键决策

### 1. NET2 是最优模型 (非 NET4)
- **发现**: 噪声估计器在所有配置下均降低性能
- **NET2** = 共享特征提取器 + 动态卷积(DynamicConv2d) + 偏振注意力(PolarAttention)
- **证据**: NET2 DoLP=28.81 vs NET4 DoLP=28.62, NET2 更轻(2.57M vs 2.58M)更快(11.70ms vs 13.24ms)
- **论文策略**: NET2 作为最终 PGPD-Net，噪声估计器作为"尝试但无效"的消融项

### 2. 损失函数配置
- alpha=0.5 (DoLP 损失权重), gamma=0.0 (噪声估计损失禁用)
- 噪声估计器没有监督信号 → 未学到有用信息

### 3. 数据路径
- 原电脑: `G:\zzz\2\` (GTX 1650)
- 当前电脑: `D:\LZ\zhoukai_paper\2\` (RTX 4070)
- dataset_info.json 中路径需自动映射
