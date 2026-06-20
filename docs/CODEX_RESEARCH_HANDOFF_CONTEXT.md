# Codex Research Handoff Context

This file is a compact handoff package for another Codex/AI agent to quickly understand the current research state, experiment protocol, manuscript direction, and known caveats.

Last updated: 2026-06-19

## 1. Project Identity

- Main current research line: polarization-relation-preserving restoration for metallic HDR DoFP polarization imaging.
- Main method name used in the current OLT-oriented manuscript: `Stokes-gain`.
- Main dataset name used in the manuscript narrative: `Metal-HDR-Pol500`.
- Main benchmark subset used for quantitative experiments: `public300`.
- Target journal: `Optics & Laser Technology (OLT)`.
- Public code/data repository intended for manuscript:
  - `https://github.com/xiaoluoqiu10-ui/StokesGain-Polarization-Restoration`

Important: There is also an older PGPD-Net denoising manuscript line in `D:\LZ\zhoukai_test\OLEN_PGPD_LaTeX`. Do not mix the two narratives unless explicitly requested.

## 2. Current User-Preferred Canonical Experiment Version

Use the following experiment version as the main benchmark unless the user explicitly changes it.

- Protocol: `metal_public300_fusedref_final`
- Dataset: metallic multi-exposure DoFP polarization data.
- Benchmark subset: `public300`.
- Input exposure: `e000053`.
- Reference: fused multi-exposure reference, generated for train/val/test.
- Split: group-level train/val/test.
- Split sizes: `210 / 45 / 45` groups for public300.
- Seeds: `42 / 7 / 123`.
- Test samples per split: `1800`.
- Methods in the main table:
  - `Raw`
  - `Raw-oracle-gain`
  - `DnCNN`
  - `NAFNet-small-local`
  - `NET2-Stokes`
  - `Stokes-gain`
- DnCNN note: use stability-audited `e6s600 lr=5e-5` checkpoint results, not the earlier unstable short-training DnCNN table.

## 3. Canonical Result Files

Primary result files:

- Main summary:
  - `experiments/major_revision/results/metal_public300_fusedref_final_experiment_summary.md`
- Main table, three-split mean:
  - `experiments/major_revision/results/metal_public300_fusedref_final_main_table_3split_mean.csv`
- Main table by seed:
  - `experiments/major_revision/results/metal_public300_fusedref_final_main_table_by_seed.csv`
- Stokes mechanism table:
  - `experiments/major_revision/results/metal_public300_fusedref_final_stokes_mechanism_table.csv`
- Expert audit:
  - `experiments/major_revision/results/metal_public300_fusedref_experimental_expert_audit.md`
- Reference protocol audit:
  - `experiments/major_revision/results/fused_reference_protocol_audit.md`
- Completion checklist:
  - `experiments/major_revision/results/experiment_completion_checklist.md`

Earlier or supplementary results should not silently overwrite the canonical final version.

## 4. Main Quantitative Results

Three-split mean +/- seed-level SD:

| Method | DoLP PSNR | DoLP SSIM | AoP RMSE | AoP RMSE DoLP>0.05 | S1 flip | S2 flip | S1/S2 MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw | 8.77 +/- 0.06 | 0.153 +/- 0.001 | 33.70 +/- 0.40 deg | 43.30 +/- 0.90 deg | 41.66 +/- 0.59% | 38.98 +/- 0.41% | 0.003448 +/- 0.000089 |
| Raw-oracle-gain | 8.81 +/- 0.06 | 0.166 +/- 0.003 | 32.97 +/- 0.50 deg | 43.30 +/- 0.90 deg | 39.91 +/- 0.43% | 37.15 +/- 0.04% | 0.004341 +/- 0.000046 |
| DnCNN | 13.97 +/- 0.75 | 0.124 +/- 0.014 | 46.80 +/- 9.47 deg | 47.90 +/- 2.25 deg | 71.01 +/- 3.14% | 75.13 +/- 3.98% | 0.003729 +/- 0.000518 |
| NAFNet-small-local | 11.72 +/- 2.67 | 0.042 +/- 0.024 | 57.46 +/- 1.94 deg | 53.21 +/- 2.18 deg | 69.57 +/- 1.07% | 78.41 +/- 2.55% | 0.003445 +/- 0.000378 |
| NET2-Stokes | 12.88 +/- 2.27 | 0.091 +/- 0.024 | 69.24 +/- 0.47 deg | 58.72 +/- 0.91 deg | 82.01 +/- 0.56% | 80.94 +/- 1.94% | 0.003752 +/- 0.000619 |
| Stokes-gain | 13.29 +/- 0.26 | 0.198 +/- 0.010 | 32.39 +/- 0.57 deg | 43.31 +/- 0.90 deg | 38.47 +/- 0.53% | 35.80 +/- 0.21% | 0.002516 +/- 0.000376 |

## 5. Interpretation for Manuscript

Use this framing:

1. Metallic surfaces often have strong specular reflection and HDR response. Low exposure is needed to avoid highlight saturation while preserving measurable polarization-difference information.
2. Low exposure weakens the four DoFP polarization-angle signals and makes Stokes-derived DoLP/AoP sensitive to channel inconsistency.
3. Free-output deep restoration models can improve DoLP PSNR, but may rewrite the Stokes components and increase AoP error.
4. Stokes-gain is a constrained Stokes-domain positive-gain restoration method. It improves DoLP PSNR over raw input while better preserving AoP and S1/S2 sign relationships than free-output deep models.
5. The central claim is not "best denoising network"; it is "polarization-relation-preserving restoration for HDR metallic DoFP imaging."

Avoid overclaiming:

- Do not claim Stokes-gain universally recovers true AoP.
- Do not claim the fused reference is an absolute physical ground truth.
- Do not describe `NAFNet-small-local` as official full NAFNet.
- Do not include PDRDN/CARDN as primary reproduced baselines unless reliable same-protocol code/weights are available.

## 6. Fused Reference Protocol

Why fused reference replaced single high exposure:

- Single high exposure `e003000` had severe bad-cell/saturation risk.
- Fused reference substantially reduced bad-cell ratio.

Reference quality across three splits:

| Seed | e003000 bad-cell ratio | Fused bad-cell ratio | Changed cell ratio | AoP fused vs e003000, DoLP>0.05 |
|---:|---:|---:|---:|---:|
| 42 | 0.7154 | 0.000103 | 0.1384 | 15.60 deg |
| 7 | 0.7172 | 0.000090 | 0.1367 | 15.62 deg |
| 123 | 0.7165 | 0.000108 | 0.1378 | 15.66 deg |

Report this as a protocol audit. In the manuscript, fused reference can be called a multi-exposure reference, but be careful not to call it absolute physical ground truth.

## 7. Stokes-Gain Ablation

Seed-42 ablation under fused-reference protocol:

| Variant | Design | DoLP PSNR | DoLP SSIM | AoP RMSE | AoP RMSE DoLP>0.05 | S1 flip | S2 flip | S1/S2 MSE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Raw | raw low-exposure input | 8.711 | 0.153 | 33.686 | 43.510 | 0.415 | 0.385 | 0.003404 |
| NET2-Stokes | free four-angle output | 14.233 | 0.106 | 69.770 | 58.551 | 0.816 | 0.788 | 0.003721 |
| Shared-angle-gain | `I_theta_hat = g * I_theta` | 8.768 | 0.178 | 32.680 | 43.508 | 0.390 | 0.360 | 0.004015 |
| Independent-S1S2-gain | S0 gain + independent positive S1/S2 gains | 12.983 | 0.190 | 32.640 | 43.510 | 0.389 | 0.360 | 0.002517 |
| Stokes-gain | S0 gain + shared positive S1/S2 gain | 13.054 | 0.206 | 32.653 | 43.511 | 0.390 | 0.360 | 0.002082 |

Interpretation:

- Free four-angle output can produce high DoLP PSNR but severely damages AoP and sign consistency.
- Positive Stokes-domain gain constraints reduce the freedom to rewrite S1/S2 signs.
- Shared positive S1/S2 gain is the most conservative relation-preserving design.

## 8. Completed Experiment List

- Metal data scan and manifest generation:
  - `metal_group_manifest.csv`
  - `metal_exposure_manifest.csv`
- Exposure quality screening and pairing.
- Fused reference generation for public300 train/val/test under seeds 42/7/123.
- Main multi-split training/evaluation:
  - Raw
  - Raw-oracle-gain
  - DnCNN
  - NAFNet-small-local
  - NET2-Stokes
  - Stokes-gain
- DnCNN stability audit:
  - replaced earlier unstable short-training DnCNN with `e6s600 lr=5e-5`.
- Stokes-gain ablation on seed42.
- Stokes mechanism analysis:
  - S1/S2 sign flip
  - S1/S2 vector MSE
  - AoP RMSE
  - DoLP-thresholded AoP RMSE
  - cosine/invalid-DoLP analyses in supplementary result files.
- Efficiency comparison:
  - `metal_public300_fusedref_model_efficiency_table.csv`
- Qualitative panels and failure/representative cases:
  - result directories under `experiments/major_revision/results/*qualitative_panels`
  - `metal_public300_fusedref_retrained_seed42_representative_failure_cases.csv`

## 9. Paper/Figure State

Current OLT Stokes-gain manuscript files:

- `olt_paper/stokes_gain_olt.tex`
- `olt_paper/stokes_gain_olt.pdf`
- `olt_paper/OLT_polished_title_abstract_framework.md`
- `olt_paper/stokes_gain_title_abstract_framework_preferred_protocol.md`

Important figure plan:

- Fig. 1: real exposure/HDR motivation using true dataset images.
- Fig. 2: macro pipeline/problem mechanism.
- Fig. 3: Stokes-gain network/method structure.
- Fig. 4: dataset batch montage.
- Fig. 5: decoded four-channel real data matrix.
- Fig. 6: quantitative data plots.
- Fig. 7: qualitative comparison: Raw / DnCNN / NAFNet / NET2 / Stokes-gain / Reference with I0, DoLP, AoP, AoP error.

Generated real-source figure panels:

- `olt_paper/stokes_gain_figures/fig2_fig3_frame19_G574_cam1_real_source_panels`
- `olt_paper/stokes_gain_figures/fig2_fig3_frame19_G574_cam1_PPT_panels`

User preference:

- Real object images in figures should come from the actual dataset.
- Use `19.bmp` or full-white projection frames where possible because they are visually cleaner than fringe frames.
- Avoid putting all panels into one huge combined image when the user wants PPT-editable assets; provide separate panel images in a folder.

## 10. Older PGPD-Net Manuscript Line

There is a separate PGPD-Net denoising manuscript in:

- `D:\LZ\zhoukai_test\OLEN_PGPD_LaTeX`

Important files:

- `main.tex`
- `main.pdf`
- `main_from_chinese_draft.tex`
- `main_from_chinese_draft.pdf`
- `main_cn_translation.tex`
- `main_cn_translation.pdf`
- `references.bib`

Older PGPD-Net result summary:

- Dataset: 128 paired noisy-clean polarization image groups.
- Input: four polarization angle images.
- Model: PGPD-Net with shared feature extractor, dynamic convolution, polarization attention, spatial noise-feature branch, and four angle decoders.
- DoLP reconstructed by Stokes formula.
- Reported DoLP result: 28.05 dB PSNR, 0.792 SSIM.
- Compared against BM3D, K-SVD, PDRDN.
- Net variants:
  - Net-1: no dynamic convolution/attention, no noise-feature branch.
  - Net-2: dynamic convolution + polarization attention.
  - Net-3: noise-feature branch.
  - Net-4/PGPD-Net: all modules.

Do not confuse this older PGPD-Net paper with the newer Stokes-gain metallic HDR paper.

## 11. Known Open Issues

For Stokes-gain paper:

- Need final decision whether the manuscript uses `Metal-HDR-Pol500` as released dataset while `public300` remains the benchmark subset. Current preference: mention 500 groups/code released, but main quantitative protocol uses public300 benchmark.
- Need keep wording precise around fused reference.
- Need ensure all figures use real dataset images where claimed.
- Need avoid saying Stokes-gain "recovers true AoP"; say it reduces additional network-induced AoP distortion or preserves Stokes-domain relations.
- Need decide whether the final main table uses stable-DnCNN final version; latest user preference says yes.

For PGPD-Net paper:

- User prefers the original Chinese draft structure.
- `main_from_chinese_draft.tex/pdf` was created to preserve that structure.
- References [58]-[77] were added to `references.bib`.
- Some older reference placeholders may remain in other versions, but `main_from_chinese_draft.tex` uses verified refs in the introduction.

## 12. Recommended Instructions for Next Codex

When continuing:

1. Ask which paper line is active: Stokes-gain OLT metallic HDR paper or PGPD-Net denoising paper.
2. If working on Stokes-gain, use `metal_public300_fusedref_final_*` as the main evidence.
3. If working on PGPD-Net, preserve the user's Chinese draft structure and avoid large narrative rewrites.
4. Before writing conclusions, verify numbers from CSV/JSON result files, not memory.
5. Do not invent baselines, DOIs, data splits, figures, or unsupported claims.
