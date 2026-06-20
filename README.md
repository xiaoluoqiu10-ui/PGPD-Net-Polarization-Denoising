# PGPD-Net: Physics-Guided DoFP Polarization Image Denoising

This repository provides the code, model configuration, representative dataset files, and experiment summaries for the PGPD-Net polarization image denoising manuscript.

## Project Scope

PGPD-Net is designed for real paired division-of-focal-plane (DoFP) polarization image denoising. The input consists of four polarization-angle images:

- `I0`
- `I45`
- `I90`
- `I135`

The restored angle images are used to reconstruct Stokes parameters and DoLP:

- `S0 = (I0 + I45 + I90 + I135) / 2`
- `S1 = I0 - I90`
- `S2 = I45 - I135`
- `DoLP = sqrt(S1^2 + S2^2) / (S0 + epsilon)`

The manuscript selects `NET2` as the final PGPD-Net configuration because it achieved the best overall denoising and DoLP reconstruction performance in the ablation study.

## Repository Structure

```text
code/       Core model, training, and evaluation scripts
data/       Dataset index and one real sample scene for format inspection
weights/    Released NET2 checkpoint
results/    Main experimental summaries and machine-readable result files
docs/       Project notes and reproducibility handoff documents
```

## Model Variants

| Model | Configuration | Manuscript role |
|---|---|---|
| NET1 | Baseline network without DynamicConv, PolarAttention, or noise estimation | Ablation baseline |
| NET2 | DynamicConv + PolarAttention, without noise-estimation supervision | Final PGPD-Net model |
| NET3 | Noise-estimation branch only | Ablation variant |
| NET4 | DynamicConv + PolarAttention + noise-estimation branch | Full-module diagnostic |

The final experimental conclusion is that the noise-estimation branch did not improve the paired real-denoising benchmark under the current training protocol. Therefore, `NET2` is used as the final model.

## Dataset

The processed dataset used in the current release is organized by scene. Each scene contains four noisy exposure/noise settings and one clean reference for each angle channel:

```text
scene_id/
  I0_gt.bmp
  I45_gt.bmp
  I90_gt.bmp
  I135_gt.bmp
  Iphotodolp_gt.bmp
  I0_noise_500.bmp
  I45_noise_500.bmp
  ...
  I135_noise_4000.bmp
```

The included `data/dataset_info.json` records the dataset index. Because the full image dataset is relatively large, this repository currently includes one real sample scene for format verification. The full dataset can be added through GitHub Releases, Git LFS, or an external archival link before manuscript submission.

## Main Results

The main result files are available in `results/`:

- `01_EXPERIMENTAL_RESULTS.md`
- `full_table.json`
- `bm3d_results.json`
- `material_results.json`
- `exp4_model_analysis.json`

Key result from the ablation benchmark:

| Model | DoLP PSNR | DoLP SSIM | AoP RMSE |
|---|---:|---:|---:|
| NET1 | 28.30 dB | 0.7861 | 8.65 deg |
| NET2 | 28.81 dB | 0.7976 | 8.62 deg |
| NET3 | 28.01 dB | 0.7981 | 8.56 deg |
| NET4 | 28.62 dB | 0.7886 | 8.92 deg |

BM3D comparison:

| Method | DoLP PSNR | DoLP SSIM | AoP RMSE |
|---|---:|---:|---:|
| BM3D | 24.19 dB | 0.6869 | 12.07 deg |
| PGPD-Net (NET2) | 28.81 dB | 0.7976 | 8.62 deg |

## Basic Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Run model evaluation from the repository root after placing the full dataset under `data/full_dataset` or updating the dataset path in the scripts:

```bash
cd code
python exp_all_models_aop.py
python exp_bm3d_comparison.py
python exp_material_analysis.py
python exp4_model_analysis.py
```

Some scripts were originally developed on a local Windows workstation. If a script contains an old path such as `G:\zzz\2`, replace it with the current dataset path.

## Data and Code Availability

This repository is intended to accompany the PGPD-Net manuscript. The source code, representative data format, model checkpoint, and result summaries are provided for review and reproducibility. The full dataset release should be attached through a stable release asset or archival link when the paper is submitted.

## License

The code is released under the MIT License. The dataset is released for academic research use only; see `DATASET_LICENSE.md`.
