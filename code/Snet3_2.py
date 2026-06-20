"""
文件名：Snet3_2.py
创建时间：2026-06-10
原始实验作者：罗哲
运行版整理：Codex

代码作用：
    本文件定义论文中 PGPD-Net 的网络结构和损失函数。
    它是不修改原始 Snet3_2-最终的.py 的当前运行副本。

论文对应点：
    该网络围绕“真实 DoFP 偏振图像去噪”和“DoLP 物理一致性约束”展开。
    网络输入四个偏振角度图像 I0/I45/I90/I135，输出四个去噪角度图，
    并根据 Stokes/DoLP 物理公式计算 DoLP，而不是单独任意预测 DoLP。

输入：
    I0, I45, I90, I135，形状通常为 [B, 1, H, W]，数值范围为 [0, 1]。

输出：
    一个字典，包含：
    - I0/I45/I90/I135：四个去噪后的偏振角度图；
    - dolp：由四个去噪角度图计算得到的 DoLP；
    - noise_feature：网络估计的噪声特征图。

主要模块：
    1. DynamicConv2d：动态卷积，根据输入内容调整卷积专家权重。
    2. PolarFeatureExtractor：四个偏振角共享的特征提取器。
    3. PolarAttention：对四角度偏振特征做注意力加权。
    4. NoiseLevelEstimator：从四角度噪声输入估计空间噪声特征。
    5. PolarDenoiseNet：完整 PGPD-Net 主网络。
    6. PolarLossWithNoiseEstimation：角度重建损失 + DoLP 物理一致性损失。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# 作用：动态卷积层。它包含多个卷积专家，并根据输入图像自动生成专家权重。
class DynamicConv2d(nn.Module):
    """动态卷积：让卷积核根据不同图像内容自适应变化。"""

    # 作用：初始化多个卷积专家和多尺度池化门控网络。
    def __init__(self, in_channels, out_channels, kernel_size, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        self.conv = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
            for _ in range(num_experts)
        ])
        # 多尺度 gate：分别用 1x1, 2x2, 4x4 池化，然后拼接
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(4)
        self.fc = nn.Sequential(
            nn.Linear(in_channels * (1 + 4 + 16), 64),  # 1^2 + 2^2 + 4^2 = 21 * in_channels
            nn.ReLU(),
            nn.Linear(64, num_experts),
            nn.Softmax(dim=1)
        )

    # 作用：前向传播。先计算每个专家卷积的权重，再把多个专家输出加权求和。
    def forward(self, x):
        # 提取多尺度特征
        b, c, h, w = x.shape
        feat1 = self.pool1(x).view(b, -1)          # [B, c]
        feat2 = self.pool2(x).view(b, -1)          # [B, c*4]
        feat3 = self.pool3(x).view(b, -1)          # [B, c*16]
        feat = torch.cat([feat1, feat2, feat3], dim=1)  # [B, c*21]
        weights = self.fc(feat)                     # [B, num_experts]
        out = 0
        for i, conv in enumerate(self.conv):
            out += weights[:, i:i+1, None, None] * conv(x)
        return out


# 作用：残差块。用于在不改变特征尺寸的情况下增强非线性表达能力。
class ResBlock(nn.Module):
    """残差块"""

    # 作用：初始化两层 3x3 卷积，用于学习残差信息。
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    # 作用：输出输入特征与卷积残差之和，帮助网络更稳定地训练。
    def forward(self, x):
        return x + self.conv(x)


# 作用：单角度共享特征提取器。四个偏振角度共用这套参数，减少参数量并增强一致性。
class PolarFeatureExtractor(nn.Module):
    """提取单个偏振角度特征（共享版本）"""

    # 作用：初始化动态卷积、普通卷积和残差块。
    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()
        self.conv1 = DynamicConv2d(in_channels, base_channels, 3)
        self.conv2 = nn.Conv2d(base_channels, base_channels * 2, 3, padding=1)
        self.conv3 = nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1)
        self.res_blocks = nn.Sequential(
            ResBlock(base_channels * 4),
            ResBlock(base_channels * 4),
            ResBlock(base_channels * 4)
        )

    # 作用：把单个偏振角图像转换成高维特征图。
    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        return self.res_blocks(x3)


# 作用：偏振注意力模块。它利用四个角度之间的关系，给不同角度特征分配权重。
class PolarAttention(nn.Module):
    """偏振感知的注意力机制"""

    # 作用：初始化 1x1 卷积，用于生成四个角度对应的注意力图。
    def __init__(self, channels):
        super().__init__()
        self.attention_conv = nn.Conv2d(channels * 4, channels * 4, 1)

    # 作用：融合四角度特征后生成注意力，再分别作用到各角度特征上。
    def forward(self, f0, f45, f90, f135):
        batch, c, h, w = f0.shape
        combined = torch.cat([f0, f45, f90, f135], dim=1)
        attention_weights = torch.sigmoid(self.attention_conv(combined))
        a0, a45, a90, a135 = torch.split(attention_weights, c, dim=1)
        return f0 * a0, f45 * a45, f90 * a90, f135 * a135


# 作用：偏振特征融合模块。先做偏振注意力，再压缩融合为统一特征。
class PolarFusionModule(nn.Module):
    """融合四个角度的特征"""

    # 作用：初始化偏振注意力和 1x1 特征压缩卷积。
    def __init__(self, in_channels):
        super().__init__()
        self.polar_attention = PolarAttention(in_channels)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels * 2, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels * 2, in_channels, 1)
        )

    # 作用：把 I0/I45/I90/I135 的特征融合成一个统一偏振特征。
    def forward(self, f0, f45, f90, f135):
        att_f0, att_f45, att_f90, att_f135 = self.polar_attention(f0, f45, f90, f135)
        fused = torch.cat([att_f0, att_f45, att_f90, att_f135], dim=1)
        return self.fusion_conv(fused)


# 作用：噪声估计分支。它直接从四个噪声偏振角图像中估计空间噪声特征。
class NoiseLevelEstimator(nn.Module):
    """输出空间噪声特征图 (保持空间维度)"""

    # 作用：初始化轻量卷积网络，输出与输入同分辨率的噪声特征图。
    def __init__(self, in_channels=4, out_channels=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, out_channels, 3, padding=1)   # 输出 [B, out_channels, H, W]
        )

    # 作用：把四角度图像在通道维拼接，并输出噪声特征。
    def forward(self, I0, I45, I90, I135):
        x = torch.cat([I0, I45, I90, I135], dim=1)   # [B, 4, H, W]
        noise_feature_map = self.conv(x)              # [B, out_channels, H, W]
        return noise_feature_map




# 作用：完整 PGPD-Net 主网络。输入四角度图像，输出四角度去噪图和 DoLP。
class PolarDenoiseNet(nn.Module):
    """主网络：共享特征提取器 + 输入级噪声估计，DOLP 由预测角度图计算"""

    # 作用：初始化共享特征提取、偏振融合、噪声估计和四个独立解码器。
    def __init__(self, base_channels=32, noise_feat_dim=32):
        super().__init__()

        # 共享特征提取器（四个角度共用）
        self.shared_branch = PolarFeatureExtractor(1, base_channels)

        # 偏振特征融合
        self.fusion = PolarFusionModule(base_channels * 4)

        # 输入级噪声估计器（直接从原始噪声图估计）
        self.noise_estimator = NoiseLevelEstimator(in_channels=4, out_channels=noise_feat_dim)

        # 四个角度的解码器（独立，输出不同角度的去噪图）
        self.decoder = nn.ModuleList([
            self._make_decoder(base_channels * 4 + noise_feat_dim) for _ in range(4)
        ])

        # 【修改】移除 DOLP 重建分支，DOLP 将在 forward 中由角度图计算得到

    # 作用：构建单个偏振角度的解码器，将融合特征还原为一张去噪图。
    def _make_decoder(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.ReLU(),
            ResBlock(in_channels//2),
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1),
            nn.ReLU(),
            ResBlock(in_channels // 4),
            nn.Conv2d(in_channels // 4, 1, 3, padding=1),
            nn.Sigmoid()
        )

    # 作用：完整前向传播。先去噪四角度图，再根据物理公式计算 DoLP。
    def forward(self, I0, I45, I90, I135):
        # 1. 先估计噪声特征（从原始噪声图像）
        noise_feature_map = self.noise_estimator(I0, I45, I90, I135)  # [B, noise_feat_dim, H, W]

        # 2. 特征提取（共用同一套参数）
        f0 = self.shared_branch(I0)
        f45 = self.shared_branch(I45)
        f90 = self.shared_branch(I90)
        f135 = self.shared_branch(I135)

        # 3. 特征融合
        fused = self.fusion(f0, f45, f90, f135)  # [B, base_channels*4, H, W]

        # 4. 将噪声特征拼接到融合特征上
        fused_with_noise = torch.cat([fused, noise_feature_map], dim=1)  # 沿通道拼接

        # 5. 解码得到四个去噪角度图
        I0_denoised = self.decoder[0](fused_with_noise)
        I45_denoised = self.decoder[1](fused_with_noise)
        I90_denoised = self.decoder[2](fused_with_noise)
        I135_denoised = self.decoder[3](fused_with_noise)

        # 【新增】根据物理公式计算 DOLP
        S0 = (I0_denoised + I90_denoised + I45_denoised + I135_denoised) / 2
        S0 = torch.clamp(S0, min=1e-6)#防止S0过小
        S1 = I0_denoised - I90_denoised
        S2 = I45_denoised - I135_denoised
        dolp_pred = torch.sqrt(S1 ** 2 + S2 ** 2 ) / (S0)
        dolp_pred = torch.clamp(dolp_pred, 0, 1)  # 确保在合理范围

        return {
            'I0': I0_denoised,
            'I45': I45_denoised,
            'I90': I90_denoised,
            'I135': I135_denoised,
            'dolp': dolp_pred,
            'noise_feature': noise_feature_map
        }


# =============================== 损失函数 ===============================

# 作用：训练损失。约束四角度强度图重建，并约束由物理公式计算的 DoLP。
class PolarLossWithNoiseEstimation(nn.Module):
    """修改后的损失函数：仅包含角度重建损失、DOLP直接损失和可选的噪声估计损失"""

    # 作用：初始化 L1/MSE 损失及 DoLP、噪声估计的权重。
    def __init__(self, alpha=0.1, gamma=0.01):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.alpha = alpha
        self.gamma = gamma

    # 作用：计算总损失，并分别返回角度损失、DoLP 损失和可选噪声损失。
    def forward(self, pred, target, noise_level_gt=None):
        # 1. 角度重建损失
        angle_loss = 0
        for angle in ['I0', 'I45', 'I90', 'I135']:
            angle_loss += self.l1_loss(pred[angle], target[f'{angle}_gt'])

        # 2. DOLP直接损失（由角度计算得到的 DOLP 与真值比较）
        dolp_loss = self.l1_loss(pred['dolp'], target['dolp_gt'])

        # 3. 噪声水平估计损失（可选，若无真值则忽略）
        noise_loss = 0
        if noise_level_gt is not None:
            noise_feature = pred['noise_feature']
            noise_pred = noise_feature.mean(dim=1)  # 简单映射为标量
            noise_loss = self.mse_loss(noise_pred, noise_level_gt)

        total_loss = angle_loss + self.alpha * dolp_loss + self.gamma * noise_loss

        return {
            'total': total_loss,
            'angle': angle_loss,
            'dolp': dolp_loss,
            'noise': noise_loss
        }


# =============================== 数据加载 ===============================

# 作用：一个简单数据集类，适合已经加载到内存的张量数据。
class PolarDataset(torch.utils.data.Dataset):
    """内存版数据集：直接从 noisy_images 和 clean_images 中取样。"""

    # 作用：保存噪声图和干净真值图数组。
    def __init__(self, noisy_images, clean_images):
        """
        noisy_images: [N, 4, H, W] 四个噪声角度图
        clean_images: [N, 5, H, W] 四个干净角度图 + DOLP真值
        """
        self.noisy = noisy_images
        self.clean = clean_images

    # 作用：返回一个样本，包括四角度噪声输入和五通道真值。
    def __getitem__(self, idx):
        noisy = self.noisy[idx]
        clean = self.clean[idx]
        return {
            'noisy': noisy,
            'I0_gt': clean[0:1],
            'I45_gt': clean[1:2],
            'I90_gt': clean[2:3],
            'I135_gt': clean[3:4],
            'dolp_gt': clean[4:5]
        }

    # 作用：返回样本总数。
    def __len__(self):
        return len(self.noisy)


# =============================== 训练循环示例 ===============================

# 作用：一个简化训练 epoch 示例。正式训练使用 train3-2_final_run.py。
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        noisy = batch['noisy'].to(device)
        targets = {k: v.to(device) for k, v in batch.items() if k != 'noisy'}

        outputs = model(noisy[:, 0:1], noisy[:, 1:2], noisy[:, 2:3], noisy[:, 3:4])

        loss_dict = criterion(outputs, targets)
        loss = loss_dict['total']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
