import torch
import torch.nn as nn
import torch.nn.functional as F
"""
    在初代版本上修改了：
    1.共享特征提取器
    只用了一个 self.shared_branch，四个角度复用同一套卷积核，参数量减少 75%，特征更对齐。

    2.输入级噪声估计
    NoiseLevelEstimator 改为接收四个原始噪声角度图（拼成 4 通道）直接估计噪声特征，逻辑因果正确，避免循环依赖。

    3.前向流程优化
    噪声特征在特征提取之前计算，先有噪声特征，后有融合特征，符合“噪声水平由输入决定”的物理直觉。

    4.保持灵活性
    损失函数中的噪声监督仍保留，但默认权重 gamma=0.01，若无噪声真值可设 gamma=0 关闭。设为零相当于删除这一损失

    5.【2025-02-14 修改】DOLP 改为由预测的四角度图像通过物理公式计算，不再通过特征图重建。
       移除 DOLPReconstruction 分支，损失函数中去掉 consistency_loss，仅保留角度损失和 DOLP 直接损失。
"""

class ResBlock(nn.Module):
    """残差块"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, x):
        return x + self.conv(x)


class PolarFeatureExtractor(nn.Module):
    """提取单个偏振角度特征（共享版本）"""
    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(base_channels, base_channels * 2, 3, padding=1)
        self.conv3 = nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1)
        self.res_blocks = nn.Sequential(
            ResBlock(base_channels * 4),
            ResBlock(base_channels * 4),
            ResBlock(base_channels * 4)
        )

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        return self.res_blocks(x3)


class PolarAttention(nn.Module):
    """偏振感知的注意力机制"""
    def __init__(self, channels):
        super().__init__()
        self.attention_conv = nn.Conv2d(channels * 4, channels * 4, 1)

    def forward(self, f0, f45, f90, f135):
        batch, c, h, w = f0.shape
        combined = torch.cat([f0, f45, f90, f135], dim=1)
        attention_weights = torch.sigmoid(self.attention_conv(combined))
        a0, a45, a90, a135 = torch.split(attention_weights, c, dim=1)
        return f0 * a0, f45 * a45, f90 * a90, f135 * a135


class PolarFusionModule(nn.Module):
    """融合四个角度的特征"""
    def __init__(self, in_channels):
        super().__init__()
        self.polar_attention = PolarAttention(in_channels)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels * 2, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels * 2, in_channels, 1)
        )

    def forward(self, f0, f45, f90, f135):
        att_f0, att_f45, att_f90, att_f135 = self.polar_attention(f0, f45, f90, f135)
        fused = torch.cat([att_f0, att_f45, att_f90, att_f135], dim=1)
        return self.fusion_conv(fused)


class NoiseLevelEstimator(nn.Module):
    """输入级噪声水平估计：直接从四个噪声角度图估计噪声特征向量"""
    def __init__(self, in_channels=4, out_channels=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(32, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, I0, I45, I90, I135):
        # 拼接四个角度图 → [B, 4, H, W]
        x = torch.cat([I0, I45, I90, I135], dim=1)
        feat = self.conv(x)          # [B, 32, 1, 1]
        feat = feat.flatten(1)       # [B, 32]
        noise_feature = self.fc(feat) # [B, out_channels]
        return noise_feature


class PolarDenoiseNet(nn.Module):
    """主网络：共享特征提取器 + 输入级噪声估计，DOLP 由预测角度图计算"""
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

    def _make_decoder(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, I0, I45, I90, I135):
        # 1. 先估计噪声特征（从原始噪声图像）
        noise_feature = self.noise_estimator(I0, I45, I90, I135)  # [B, noise_feat_dim]

        # 2. 特征提取（共用同一套参数）
        f0 = self.shared_branch(I0)
        f45 = self.shared_branch(I45)
        f90 = self.shared_branch(I90)
        f135 = self.shared_branch(I135)

        # 3. 特征融合
        fused = self.fusion(f0, f45, f90, f135)  # [B, base_channels*4, H, W]

        # 4. 将噪声特征拼接到融合特征上
        B, C, H, W = fused.shape
        noise_feature_expanded = noise_feature.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        fused_with_noise = torch.cat([fused, noise_feature_expanded], dim=1)

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
            'noise_feature': noise_feature
        }


# =============================== 损失函数 ===============================

class PolarLossWithNoiseEstimation(nn.Module):
    """修改后的损失函数：仅包含角度重建损失、DOLP直接损失和可选的噪声估计损失"""
    def __init__(self, alpha=0.1, gamma=0.01):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.alpha = alpha
        self.gamma = gamma

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

class PolarDataset(torch.utils.data.Dataset):
    def __init__(self, noisy_images, clean_images):
        """
        noisy_images: [N, 4, H, W] 四个噪声角度图
        clean_images: [N, 5, H, W] 四个干净角度图 + DOLP真值
        """
        self.noisy = noisy_images
        self.clean = clean_images

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

    def __len__(self):
        return len(self.noisy)


# =============================== 训练循环示例 ===============================

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