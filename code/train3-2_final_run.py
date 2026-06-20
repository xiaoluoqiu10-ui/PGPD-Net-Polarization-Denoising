"""
文件名：train3-2_final_run.py
创建时间：2026-06-10
原始实验作者：罗哲
运行版整理：Codex

代码作用：
    本脚本是不修改原始训练代码的“当前电脑可运行版”。它用于训练论文中的
    PGPD-Net 偏振图像去噪模型，对应论文核心问题：
    “真实 DoFP 偏振相机采集噪声下，如何利用四个偏振角度图像
    I0/I45/I90/I135 重建更稳定的 DoLP 图像”。

输入：
    1. 数据集索引文件：
       D:\\LZ\\zhoukai_paper\\2\\new\\data2.1\\dataset_info.json
    2. 每组数据包含四个偏振角通道 I0/I45/I90/I135。
    3. 噪声图像来自不同曝光/增益组合：
       500 ms/24 dB、1000 ms/18 dB、2000 ms/12 dB、4000 ms/6 dB。
    4. 真值图像来自 8000 ms/0 dB，并包含 DoLP 真值图。

输出：
    1. 每轮训练损失和验证 PSNR 打印到终端。
    2. checkpoint_epoch_XXX.pth：每轮模型检查点。
    3. latest_checkpoint.pth：最近一轮检查点。
    4. best_model.pth：当前验证 PSNR 最优模型。

使用方式：
    快速检查是否能跑通：
        python train3-2_final_run.py --smoke-test

    短训练 1 轮，确认完整训练流程：
        python train3-2_final_run.py --epochs 1

    正式训练：
        python train3-2_final_run.py

说明：
    这个文件只用于当前环境复现实验。原始文件 train3-2最终的.py 不会被修改。
"""

import argparse
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn.functional as F

# 导入当前可运行副本中的最终网络结构。Snet3_2.py 由 Snet3_2-最终的.py 复制得到。
from Snet3_2 import PolarDenoiseNet, PolarLossWithNoiseEstimation

# -------------------- 配置参数 --------------------
class Config:
    patch_size = 128
    stride = 64
    target_height = 1024
    target_width = 1224
    base_channels = 32
    noise_feat_dim = 32
    batch_size = 2
    num_epochs = 35
    learning_rate = 5e-5
    weight_decay = 1e-5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    json_path = r'D:\LZ\zhoukai_paper\2\new\data2.1\dataset_info.json'
    output_dir = r'D:\LZ\zhoukai_paper\lz\runs\pgpd_train3_2_final'

    # 损失权重
    alpha = 0.5
    gamma = 0.0

    # 数据划分
    val_ratio = 0.2
    seed = 42

    # 恢复训练（从头训练时设为 False）
    resume = False
    resume_checkpoint = 'checkpoint_epoch_027.pth'
    smoke_scene = '1'

config = Config()
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
print(f"Using device: {config.device}")
if config.device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# 作用：把 dataset_info.json 中保存的旧电脑绝对路径，转换为当前电脑上的真实路径。
# 原因：原 JSON 内部仍然记录 G:\zzz\2\new\data2.1，但当前数据在 D:\LZ\zhoukai_paper\2\new\data2.1。
def resolve_image_path(path, dataset_root):
    candidate = Path(path)
    if candidate.exists():
        return str(candidate)

    path_text = str(path)
    old_roots = [
        r'G:\zzz\2\new\data2.1',
        r'G:/zzz/2/new/data2.1',
    ]
    for old_root in old_roots:
        if path_text.startswith(old_root):
            translated = Path(str(dataset_root)) / path_text[len(old_root):].lstrip(r'\/')
            if translated.exists():
                return str(translated)

    raise FileNotFoundError(
        f"图像文件不存在：{path}\n"
        f"也无法自动映射到当前数据根目录：{dataset_root}"
    )


# 作用：读取命令行参数，让不熟悉代码的使用者也能通过简单命令切换测试/训练模式。
def parse_args():
    parser = argparse.ArgumentParser(description='PGPD-Net current-machine training runner.')
    parser.add_argument('--smoke-test', action='store_true',
                        help='只读取一组数据并跑一个训练 batch，用于快速确认代码和数据能否跑通。')
    parser.add_argument('--json-path', default=config.json_path,
                        help='dataset_info.json 路径。默认使用当前数据集路径。')
    parser.add_argument('--output-dir', default=config.output_dir,
                        help='模型 checkpoint 输出目录。')
    parser.add_argument('--epochs', type=int, default=config.num_epochs,
                        help='训练轮数。正式训练默认 35；检查流程可设为 1。')
    parser.add_argument('--batch-size', type=int, default=config.batch_size,
                        help='训练 batch size。显存不足时可设为 1。')
    parser.add_argument('--cpu', action='store_true',
                        help='强制使用 CPU。一般不需要，除非 CUDA 环境异常。')
    parser.add_argument('--resume', action='store_true',
                        help='从 checkpoint 继续训练。默认从 output-dir/latest_checkpoint.pth 读取。')
    parser.add_argument('--resume-checkpoint', default=None,
                        help='指定继续训练用的 checkpoint 路径。')
    return parser.parse_args()


# 作用：把命令行参数写入全局 config，保证后面的数据集、模型和训练循环使用同一套设置。
def apply_args_to_config(args):
    config.json_path = args.json_path
    config.output_dir = args.output_dir
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.resume = args.resume
    if args.resume_checkpoint:
        config.resume_checkpoint = args.resume_checkpoint
    elif args.resume:
        config.resume_checkpoint = os.path.join(config.output_dir, 'latest_checkpoint.pth')
    if args.cpu:
        config.device = torch.device('cpu')


# 作用：打印本次运行的关键信息，方便直接判断是否读到了正确数据、是否使用 GPU、输出在哪里。
def print_run_header(mode):
    print("=" * 80)
    print(f"PGPD-Net 运行模式：{mode}")
    print(f"训练脚本：{Path(__file__).resolve()}")
    print(f"数据索引：{config.json_path}")
    print(f"输出目录：{config.output_dir}")
    print(f"设备：{config.device}")
    if config.device.type == 'cuda':
        print(f"GPU：{torch.cuda.get_device_name(0)}")
    print(f"训练轮数：{config.num_epochs}")
    print(f"Batch size：{config.batch_size}")
    print("=" * 80)


# -------------------- 数据集类（与之前相同） --------------------
class PolarPatchDataset(Dataset):
    """作用：按 scene 和噪声等级读取四角度偏振图像，并在训练阶段切成 128x128 patch。"""

    def __init__(self, json_path, scene_list, phase='train', transform=None):
        self.phase = phase
        self.transform = transform
        self.patch_size = config.patch_size
        self.stride = config.stride
        self.orig_h = config.target_height
        self.orig_w = config.target_width
        self.dataset_root = Path(json_path).resolve().parent

        with open(json_path, 'r') as f:
            all_scenes = json.load(f)

        self.samples = []
        for scene_data in all_scenes:
            scene = scene_data['scene']
            if scene not in scene_list:
                continue
            gt = scene_data['gt_images']
            noise_dict = scene_data['noise_images']
            for level, paths in noise_dict.items():
                sample = {
                    'noise_paths': [paths['I0'], paths['I45'], paths['I90'], paths['I135']],
                    'gt_paths': [gt['I0'], gt['I45'], gt['I90'], gt['I135'], gt['dolp']],
                    'noise_level': int(level)
                }
                self.samples.append(sample)

        print(f"[{phase}] 正在加载图像到内存，共 {len(self.samples)} 个样本...")
        self.noisy_images = []
        self.gt_images = []
        for idx, sample in enumerate(self.samples):
            if (idx + 1) % 10 == 0:
                print(f"[{phase}] 已加载 {idx+1}/{len(self.samples)} 个样本")
            noise_imgs = [self._read_image(p) for p in sample['noise_paths']]
            gt_imgs = [self._read_image(p) for p in sample['gt_paths']]
            self.noisy_images.append(np.stack(noise_imgs, axis=0))
            self.gt_images.append(np.stack(gt_imgs, axis=0))
        print(f"[{phase}] 图像加载完成。")

        # 统计 S0 最小值
        min_s0 = float('inf')
        for gt in self.gt_images:
            s0 = (gt[0] + gt[1] + gt[2] + gt[3]) / 2.0
            min_s0 = min(min_s0, s0.min())
        print(f"[{phase}] 全局 S0 最小值: {min_s0:.6f} (若小于 1e-4 可能导致数值不稳定)")

        if self.phase == 'train':
            self.patch_indices = []
            for idx in range(len(self.samples)):
                top_positions = list(range(0, self.orig_h - self.patch_size + 1, self.stride))
                if top_positions[-1] != self.orig_h - self.patch_size:
                    top_positions.append(self.orig_h - self.patch_size)
                left_positions = list(range(0, self.orig_w - self.patch_size + 1, self.stride))
                if left_positions[-1] != self.orig_w - self.patch_size:
                    left_positions.append(self.orig_w - self.patch_size)
                for top in top_positions:
                    for left in left_positions:
                        self.patch_indices.append((idx, top, left))
            print(f"[{phase}] 共生成 {len(self.patch_indices)} 个训练块")

    # 作用：返回训练 patch 数量或验证图像数量，供 DataLoader 迭代使用。
    def __len__(self):
        if self.phase == 'train':
            return len(self.patch_indices)
        else:
            return len(self.samples)

    # 作用：读取一个训练 patch 或一整张验证图像，并返回网络训练需要的字典。
    def __getitem__(self, idx):
        try:
            if self.phase == 'train':
                sample_idx, top, left = self.patch_indices[idx]
                noise = self.noisy_images[sample_idx]
                gt = self.gt_images[sample_idx]
                noise_patch = noise[:, top:top+self.patch_size, left:left+self.patch_size]
                gt_patch = gt[:, top:top+self.patch_size, left:left+self.patch_size]
                noise_tensor = torch.from_numpy(noise_patch).float()
                gt_tensor = torch.from_numpy(gt_patch).float()
                return {
                    'noisy': noise_tensor,
                    'I0_gt': gt_tensor[0:1],
                    'I45_gt': gt_tensor[1:2],
                    'I90_gt': gt_tensor[2:3],
                    'I135_gt': gt_tensor[3:4],
                    'dolp_gt': gt_tensor[4:5],
                    'noise_level': torch.tensor(self.samples[sample_idx]['noise_level']/1000.0, dtype=torch.float32)
                }
            else:
                noise_tensor = torch.from_numpy(self.noisy_images[idx]).float()
                gt_tensor = torch.from_numpy(self.gt_images[idx]).float()
                return {
                    'noisy': noise_tensor,
                    'I0_gt': gt_tensor[0:1],
                    'I45_gt': gt_tensor[1:2],
                    'I90_gt': gt_tensor[2:3],
                    'I135_gt': gt_tensor[3:4],
                    'dolp_gt': gt_tensor[4:5],
                    'noise_level': torch.tensor(self.samples[idx]['noise_level']/1000.0, dtype=torch.float32)
                }
        except Exception as e:
            print(f"Error loading sample index {idx}: {e}")
            return None

    # 作用：读取单通道灰度图并归一化到 [0, 1]，同时自动修正旧电脑绝对路径。
    def _read_image(self, path):
        image_path = resolve_image_path(path, self.dataset_root)
        with Image.open(image_path) as img:
            img = img.convert('L')
            arr = np.array(img, dtype=np.float32) / 255.0
        return arr


# 作用：过滤掉偶发读取失败的样本，避免一个坏样本让整个 DataLoader 崩掉。
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return {}
    return torch.utils.data.dataloader.default_collate(batch)

# -------------------- 辅助函数（高斯融合、PSNR）保持不变 --------------------
# 作用：生成二维高斯权重，用于把重叠 patch 的预测结果平滑融合成整图。
def gaussian_kernel(size, sigma=None, device='cpu'):
    if sigma is None:
        sigma = size / 3.0
    ax = torch.arange(size).float().to(device) - (size - 1.0) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / kernel.max()


# 作用：对一整张大图进行 patch 滑窗预测，并用高斯权重融合，避免 patch 边界出现明显接缝。
def predict_full_image(model, noisy_tensor, device, patch_size=128, stride=64):
    model.eval()
    with torch.no_grad():
        C, H, W = noisy_tensor.shape
        accum_I0 = torch.zeros((H, W), device=device)
        accum_I45 = torch.zeros((H, W), device=device)
        accum_I90 = torch.zeros((H, W), device=device)
        accum_I135 = torch.zeros((H, W), device=device)
        accum_dolp = torch.zeros((H, W), device=device)
        weight_sum = torch.zeros((H, W), device=device)

        gauss_kernel = gaussian_kernel(patch_size, device=device)

        top_positions = list(range(0, H - patch_size + 1, stride))
        if top_positions[-1] != H - patch_size:
            top_positions.append(H - patch_size)
        left_positions = list(range(0, W - patch_size + 1, stride))
        if left_positions[-1] != W - patch_size:
            left_positions.append(W - patch_size)

        for top in top_positions:
            for left in left_positions:
                patch = noisy_tensor[:, top:top+patch_size, left:left+patch_size].unsqueeze(0).to(device)
                outputs = model(patch[:, 0:1], patch[:, 1:2], patch[:, 2:3], patch[:, 3:4])
                I0_pred = outputs['I0'][0, 0]
                I45_pred = outputs['I45'][0, 0]
                I90_pred = outputs['I90'][0, 0]
                I135_pred = outputs['I135'][0, 0]
                dolp_pred = outputs['dolp'][0, 0]

                accum_I0[top:top+patch_size, left:left+patch_size] += I0_pred * gauss_kernel
                accum_I45[top:top+patch_size, left:left+patch_size] += I45_pred * gauss_kernel
                accum_I90[top:top+patch_size, left:left+patch_size] += I90_pred * gauss_kernel
                accum_I135[top:top+patch_size, left:left+patch_size] += I135_pred * gauss_kernel
                accum_dolp[top:top+patch_size, left:left+patch_size] += dolp_pred * gauss_kernel
                weight_sum[top:top+patch_size, left:left+patch_size] += gauss_kernel

        weight_sum = torch.clamp(weight_sum, min=1e-8)
        I0_full = accum_I0 / weight_sum
        I45_full = accum_I45 / weight_sum
        I90_full = accum_I90 / weight_sum
        I135_full = accum_I135 / weight_sum
        dolp_full = accum_dolp / weight_sum
        return I0_full, I45_full, I90_full, I135_full, dolp_full


# 作用：计算 PSNR 指标。PSNR 越高，一般说明预测图像越接近真值图像。
def compute_psnr(pred, target, max_val=1.0):
    if pred.dim() == 3:
        pred = pred.squeeze(0)
        target = target.squeeze(0)
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    psnr = 10 * torch.log10(max_val**2 / mse)
    return psnr.item()


# 作用：快速检查“数据读取、网络前向、loss、反向传播、优化器更新”是否全部正常。
# 这个函数只跑一组 scene 的一个 batch，不会保存模型，适合正式训练前先看终端输出。
def run_smoke_test():
    print_run_header('smoke-test 快速连通性测试')
    print("开始 smoke test：只使用 scene 1 的一个 batch，不进行完整训练。")

    dataset = PolarPatchDataset(config.json_path, [config.smoke_scene], phase='train')
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False,
                        num_workers=0, pin_memory=True, collate_fn=collate_fn, drop_last=True)
    batch = next(iter(loader))
    noisy = batch['noisy'].to(config.device)
    targets = {
        'I0_gt': batch['I0_gt'].to(config.device),
        'I45_gt': batch['I45_gt'].to(config.device),
        'I90_gt': batch['I90_gt'].to(config.device),
        'I135_gt': batch['I135_gt'].to(config.device),
        'dolp_gt': batch['dolp_gt'].to(config.device)
    }

    model = PolarDenoiseNet(base_channels=config.base_channels, noise_feat_dim=config.noise_feat_dim).to(config.device)
    criterion = PolarLossWithNoiseEstimation(alpha=config.alpha, gamma=config.gamma)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    outputs = model(noisy[:, 0:1], noisy[:, 1:2], noisy[:, 2:3], noisy[:, 3:4])
    loss_dict = criterion(outputs, targets, None)
    loss = loss_dict['total']
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    print("-" * 80)
    print("smoke test 结果：")
    print(f"输入 batch noisy shape: {tuple(noisy.shape)}，数值范围：{noisy.min().item():.4f} ~ {noisy.max().item():.4f}")
    print(f"I0 输出 shape: {tuple(outputs['I0'].shape)}")
    print(f"I45 输出 shape: {tuple(outputs['I45'].shape)}")
    print(f"I90 输出 shape: {tuple(outputs['I90'].shape)}")
    print(f"I135 输出 shape: {tuple(outputs['I135'].shape)}")
    print(f"DoLP 输出 shape: {tuple(outputs['dolp'].shape)}")
    print(f"噪声特征 shape: {tuple(outputs['noise_feature'].shape)}")
    print(f"total loss: {loss_dict['total'].item():.6f}")
    print(f"angle loss: {loss_dict['angle'].item():.6f}")
    print(f"dolp loss: {loss_dict['dolp'].item():.6f}")
    if config.device.type == 'cuda':
        print(f"CUDA 显存占用：{torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
    print("结论：如果你看到本行，说明数据读取、网络前向、loss、反向传播和优化器更新均已跑通。")
    print("-" * 80)


# -------------------- 主训练脚本 --------------------
# 作用：正式训练入口。它会划分训练/验证 scene，循环训练，并保存 checkpoint。
def main():
    print_run_header('正式训练')
    os.makedirs(config.output_dir, exist_ok=True)

    # 数据准备
    with open(config.json_path, 'r') as f:
        all_scenes = json.load(f)
    all_scene_ids = [str(s['scene']) for s in all_scenes]
    random.shuffle(all_scene_ids)
    split_idx = int(len(all_scene_ids) * (1 - config.val_ratio))
    train_scenes = all_scene_ids[:split_idx]
    val_scenes = all_scene_ids[split_idx:]

    print("开始创建训练数据集...")
    train_dataset = PolarPatchDataset(config.json_path, train_scenes, phase='train')
    print(f"训练数据集创建完成，共有 {len(train_dataset)} 个样本")
    print("开始创建验证数据集...")
    val_dataset = PolarPatchDataset(config.json_path, val_scenes, phase='val')
    print("验证数据集创建完成")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=0, pin_memory=True, collate_fn=collate_fn)
    print("DataLoader 创建完成")

    # 创建模型、损失函数、优化器
    model = PolarDenoiseNet(base_channels=config.base_channels, noise_feat_dim=config.noise_feat_dim).to(config.device)
    criterion = PolarLossWithNoiseEstimation(alpha=config.alpha, gamma=config.gamma)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # 学习率预热 + ReduceLROnPlateau
    warmup_epochs = 5
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 恢复训练（新模型从头训练，resume=False 时忽略）
    start_epoch = 1
    best_val_psnr = 0.0
    if config.resume:
        if os.path.isfile(config.resume_checkpoint):
            print(f"正在从 {config.resume_checkpoint} 恢复训练...")
            checkpoint = torch.load(config.resume_checkpoint, map_location=config.device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # 注意：调度器恢复需小心处理，这里简化
            if 'scheduler_state_dict' in checkpoint:
                if 'warmup_scheduler' in checkpoint['scheduler_state_dict']:
                    warmup_scheduler.load_state_dict(checkpoint['scheduler_state_dict']['warmup_scheduler'])
                if 'main_scheduler' in checkpoint['scheduler_state_dict']:
                    main_scheduler.load_state_dict(checkpoint['scheduler_state_dict']['main_scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_psnr = checkpoint['best_val_psnr']
            print(f"恢复成功，将从 epoch {start_epoch} 继续训练，当前最佳 PSNR: {best_val_psnr:.4f}")
        else:
            print(f"警告：恢复文件不存在，从头训练。")

    # 训练循环
    for epoch in range(start_epoch, config.num_epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0
        for batch in train_loader:
            if not batch:
                continue
            noisy = batch['noisy'].to(config.device)
            targets = {
                'I0_gt': batch['I0_gt'].to(config.device),
                'I45_gt': batch['I45_gt'].to(config.device),
                'I90_gt': batch['I90_gt'].to(config.device),
                'I135_gt': batch['I135_gt'].to(config.device),
                'dolp_gt': batch['dolp_gt'].to(config.device)
            }
            noise_level = batch['noise_level'].to(config.device) if config.gamma > 0 else None

            outputs = model(noisy[:,0:1], noisy[:,1:2], noisy[:,2:3], noisy[:,3:4])
            loss_dict = criterion(outputs, targets, noise_level)
            loss = loss_dict['total']

            optimizer.zero_grad()
            loss.backward()

            # 梯度检查
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isinf(param.grad).any():
                        print(f"Gradient inf in {name}")
                    if torch.isnan(param.grad).any():
                        print(f"Gradient nan in {name}")

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f}")

        if np.isnan(avg_train_loss):
            print(f"Epoch {epoch} 训练损失为 NaN，训练终止。请使用上一轮 checkpoint 恢复。")
            break

        # 验证
        model.eval()
        psnr_list = {'I0': [], 'I45': [], 'I90': [], 'I135': [], 'dolp': []}
        with torch.no_grad():
            for batch in val_loader:
                if not batch:
                    continue
                noisy = batch['noisy'][0].to(config.device)
                I0_pred, I45_pred, I90_pred, I135_pred, dolp_pred = predict_full_image(
                    model, noisy, config.device, config.patch_size, config.stride
                )
                I0_gt = batch['I0_gt'][0].to(config.device).squeeze(0)
                I45_gt = batch['I45_gt'][0].to(config.device).squeeze(0)
                I90_gt = batch['I90_gt'][0].to(config.device).squeeze(0)
                I135_gt = batch['I135_gt'][0].to(config.device).squeeze(0)
                dolp_gt = batch['dolp_gt'][0].to(config.device).squeeze(0)

                I0_pred_clip = torch.clamp(I0_pred, 0, 1)
                I45_pred_clip = torch.clamp(I45_pred, 0, 1)
                I90_pred_clip = torch.clamp(I90_pred, 0, 1)
                I135_pred_clip = torch.clamp(I135_pred, 0, 1)
                dolp_pred_clip = torch.clamp(dolp_pred, 0, 1)

                psnr_list['I0'].append(compute_psnr(I0_pred_clip, I0_gt))
                psnr_list['I45'].append(compute_psnr(I45_pred_clip, I45_gt))
                psnr_list['I90'].append(compute_psnr(I90_pred_clip, I90_gt))
                psnr_list['I135'].append(compute_psnr(I135_pred_clip, I135_gt))
                psnr_list['dolp'].append(compute_psnr(dolp_pred_clip, dolp_gt))

        avg_psnr = {k: np.mean(v) for k, v in psnr_list.items()}
        print(f"Val PSNR: I0={avg_psnr['I0']:.2f}, I45={avg_psnr['I45']:.2f}, I90={avg_psnr['I90']:.2f}, I135={avg_psnr['I135']:.2f}, DOLP={avg_psnr['dolp']:.2f}")
        avg_total_psnr = np.mean(list(avg_psnr.values()))

        # 学习率调度
        if epoch <= warmup_epochs:
            warmup_scheduler.step()
        else:
            main_scheduler.step(avg_train_loss)

        # 保存每轮 checkpoint
        epoch_checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': {
                'warmup_scheduler': warmup_scheduler.state_dict(),
                'main_scheduler': main_scheduler.state_dict()
            },
            'best_val_psnr': best_val_psnr,
        }
        torch.save(epoch_checkpoint, os.path.join(config.output_dir, f'checkpoint_epoch_{epoch:03d}.pth'))
        torch.save(epoch_checkpoint, os.path.join(config.output_dir, 'latest_checkpoint.pth'))
        print(f"Checkpoint saved for epoch {epoch:03d} (latest_checkpoint.pth updated)")

        # 保存最佳模型
        if avg_total_psnr > best_val_psnr:
            best_val_psnr = avg_total_psnr
            torch.save(epoch_checkpoint, os.path.join(config.output_dir, 'best_model.pth'))
            print("Saved best model (with full state).")

if __name__ == '__main__':
    args = parse_args()
    apply_args_to_config(args)
    if args.smoke_test:
        run_smoke_test()
    else:
        main()
