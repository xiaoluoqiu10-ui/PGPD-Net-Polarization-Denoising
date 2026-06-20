"""
多材质定量分析
根据用户提供的21场景材质标注，按材质分组统计 PSNR/SSIM
"""
import json, os, numpy as np

# ======================= 场景→材质映射 =======================
SCENE_MATERIAL = {
    '1':  'Shiny metal (quatrefoil ornament)',
    '2':  'Shiny metal (perforated part)',
    '3':  'Shiny metal (fan-blade disc)',
    '4':  'Shiny metal (thin fan-blade sheet)',
    '5':  'Shiny metal (coupling disc)',
    '6':  'Shiny metal (CNC flat part with grooves)',
    '7':  'Shiny metal (vernier caliper)',
    '8':  'Shiny metal (license plate)',
    '9':  'Shiny metal (playing card with heart cutout)',
    '10': 'Dark metal (I-beam rail sample)',
    '11': 'Mixed metal (perforated strip + support rod)',
    '12': 'Mixed metal (small wrench + dark mounting plate)',
    '13': 'Shiny metal (quatrefoil ornament v2)',
    '14': 'Black rubber (air dust blower bulb)',
    '15': 'Dark metal (phone stand)',
    '16': 'Black plastic (computer mouse)',
    '17': 'Cardboard (printed carton box)',
    '18': 'Black plastic (spray bottle)',
    '19': 'Black ceramic (mug)',
    '20': 'Dark metal (iron sword)',
    '21': 'PCB (microcontroller board)',
}

# 材料大类分组
MATERIAL_GROUPS = {
    'Shiny Metal': ['1','2','3','4','5','6','7','8','9','13'],
    'Dark Metal':  ['10','15','20'],
    'Mixed Metal': ['11','12'],
    'Rubber':      ['14'],
    'Plastic':     ['16','18'],
    'Cardboard':   ['17'],
    'Ceramic':     ['19'],
    'PCB':         ['21'],
}

# 合并为论文友好的4大类
COARSE_GROUPS = {
    'Metal (shiny)':   ['1','2','3','4','5','6','7','8','9','13'],
    'Metal (dark/coated)': ['10','11','12','15','20'],
    'Plastic/Rubber':  ['14','16','18'],
    'Other (paper/ceramic/PCB)': ['17','19','21'],
}

# ======================= 加载已有结果 =======================
with open(os.path.join(os.path.dirname(__file__), 'exp_all_results', 'full_table.json'), 'r') as f:
    data = json.load(f)

NET2 = data['NET2_dynconv_attn']

# 需要逐样本数据 — 从 per_sample 中读取
# exp_all_models_aop.py 没有保存逐样本数据，需要从 full_table 的 by_noise 聚合
# 暂时用按噪声分组的数据 + 场景材质做近似分析

print("=" * 70)
print("        Multi-Material Analysis (Scene-level mapping)")
print("=" * 70)

print("\nScene -> Material mapping:")
for sid in sorted(SCENE_MATERIAL.keys(), key=int):
    groups = []
    for gname, scenes in COARSE_GROUPS.items():
        if sid in scenes:
            groups.append(gname)
    print(f"  Scene {sid:>2}: {SCENE_MATERIAL[sid]:<50} -> {', '.join(groups)}")

print("\n--- Material Group Statistics ---")
for gname, scenes in COARSE_GROUPS.items():
    print(f"\n  [{gname}] ({len(scenes)} scenes): {', '.join(scenes)}")

# 打印按噪声水平的材质分组指导
print("\n\n--- For per-noise-level analysis, use this mapping ---")
print("To compute per-material PSNR/SSIM, run exp_all_models_aop.py")
print("with per_sample output enabled, then filter by scene ID.")
print("\nScene IDs by material group:")
for gname, scenes in COARSE_GROUPS.items():
    print(f"  {gname}: {scenes}")

# 保存映射
mapping = {
    'scene_material': SCENE_MATERIAL,
    'material_groups': MATERIAL_GROUPS,
    'coarse_groups': COARSE_GROUPS,
}
with open(os.path.join(os.path.dirname(__file__), 'exp_all_results', 'material_mapping.json'), 'w', encoding='utf-8') as f:
    json.dump(mapping, f, indent=2, ensure_ascii=False)
print("\nMapping saved to exp_all_results/material_mapping.json")
