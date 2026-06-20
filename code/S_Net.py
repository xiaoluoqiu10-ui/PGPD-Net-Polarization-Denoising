import torch
import torch.nn as nn
import torch.nn.functional as F


class PolarFeatureExtractor(nn.Module):
    """提取单个偏振角度特征"""

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
        # 应用偏振注意力
        att_f0, att_f45, att_f90, att_f135 = self.polar_attention(f0, f45, f90, f135)

        # 拼接所有特征
        fused = torch.cat([att_f0, att_f45, att_f90, att_f135], dim=1)
        return self.fusion_conv(fused)


class PolarAttention(nn.Module):
    """偏振感知的注意力机制，利用斯托克斯参数关系"""

    def __init__(self, channels):
        super().__init__()
        # 学习偏振角度间的物理关系权重
        self.attention_conv = nn.Conv2d(channels * 4, channels * 4, 1)

    def forward(self, f0, f45, f90, f135):
        batch, c, h, w = f0.shape
        # 计算偏振相关特征
        combined = torch.cat([f0, f45, f90, f135], dim=1)
        attention_weights = torch.sigmoid(self.attention_conv(combined))

        # 分割注意力权重
        a0, a45, a90, a135 = torch.split(attention_weights, c, dim=1)

        return f0 * a0, f45 * a45, f90 * a90, f135 * a135


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
class NoiseLevelEstimator(nn.Module):
    """噪声水平估计模块，输出一个特征向量"""

    def __init__(self, in_channels, out_channels=32):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.ReLU()
        )
        self.noise_level_fc = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x):
        # x: [B, C, H, W]
        pooled = self.global_pool(x)  # [B, C, 1, 1]
        features = self.conv(pooled)  # [B, out_channels, 1, 1]
        features = features.squeeze(-1).squeeze(-1)  # [B, out_channels]
        noise_feature = self.noise_level_fc(features)  # [B, out_channels]
        return noise_feature

class DOLPReconstruction(nn.Module):
    """从四个角度重建DOLP"""

    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.dolp_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, out_channels, 3, padding=1),
            nn.Sigmoid()  # DOLP范围[0,1]
        )

    def forward(self, fused_features):
        return self.dolp_conv(fused_features)


class PolarDenoiseNet(nn.Module):
    """主网络（共享特征提取器版本）"""
    def __init__(self, base_channels=32, noise_feat_dim=32):
        super().__init__()

        # === 共享同一个特征提取器（）===
        self.shared_branch = PolarFeatureExtractor(1, base_channels)

        # 偏振特征融合
        self.fusion = PolarFusionModule(base_channels * 4)

        # 噪声水平估计
        self.noise_estimator = NoiseLevelEstimator(base_channels * 4, noise_feat_dim)

        # 解码器（仍然独立，因为要输出四个不同角度图像）
        self.decoder = nn.ModuleList([
            self._make_decoder(base_channels * 4 + noise_feat_dim) for _ in range(4)
        ])

        # DOLP重建分支
        self.dolp_branch = DOLPReconstruction(base_channels * 4 + noise_feat_dim)

    def _make_decoder(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, 1, 3, padding=1)
        )

    def forward(self, I0, I45, I90, I135):
        # === 共用同一套参数，分别处理四个角度 ===
        f0 = self.shared_branch(I0)
        f45 = self.shared_branch(I45)
        f90 = self.shared_branch(I90)
        f135 = self.shared_branch(I135)

        # 后续保持不变
        fused = self.fusion(f0, f45, f90, f135)

        noise_feature = self.noise_estimator(fused)
        B, C, H, W = fused.shape
        noise_feature_expanded = noise_feature.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)

        fused_with_noise = torch.cat([fused, noise_feature_expanded], dim=1)

        I0_denoised = self.decoder[0](fused_with_noise)
        I45_denoised = self.decoder[1](fused_with_noise)
        I90_denoised = self.decoder[2](fused_with_noise)
        I135_denoised = self.decoder[3](fused_with_noise)
        dolp_pred = self.dolp_branch(fused_with_noise)

        return {
            'I0': I0_denoised,
            'I45': I45_denoised,
            'I90': I90_denoised,
            'I135': I135_denoised,
            'dolp': dolp_pred,
            'noise_feature': noise_feature
        }


    #===================================损失函数========================================

class PolarLossWithNoiseEstimation(nn.Module):
    def __init__(self, alpha=0.1, beta=0.5, gamma=0.01):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.alpha = alpha  # DOLP损失权重
        self.beta = beta  # 物理一致性损失权重
        self.gamma = gamma  # 噪声水平估计损失权重

    def forward(self, pred, target, noise_level_gt=None):
        """
        pred: 网络输出字典
        target: 包含I0_gt, I45_gt, I90_gt, I135_gt, dolp_gt
        noise_level_gt: 噪声水平真值，如果为None，则不计算噪声损失
        """
        # 1. 四个角度的重建损失
        angle_loss = 0
        for angle in ['I0', 'I45', 'I90', 'I135']:
            angle_loss += self.l1_loss(pred[angle], target[f'{angle}_gt'])

        # 2. DOLP直接损失
        dolp_loss = self.l1_loss(pred['dolp'], target['dolp_gt'])

        # 3. 物理一致性损失
        S0 = (pred['I0'] + pred['I90']) / 2
        S1 = pred['I0'] - pred['I90']
        S2 = pred['I45'] - pred['I135']
        dolp_from_angles = torch.sqrt(S1 ** 2 + S2 ** 2 + 1e-8) / (S0 + 1e-8)
        consistency_loss = self.mse_loss(dolp_from_angles, pred['dolp'])

        # 4. 噪声水平估计损失（如果有真值）
        noise_loss = 0
        if noise_level_gt is not None:
            # 这里假设噪声水平真值是一个标量，且与噪声特征向量的维度相同
            # 我们可以用MSE损失，但需要将噪声特征向量映射到一个标量
            # 或者，我们可以将噪声水平真值扩展为向量，然后计算损失
            # 这里我们简单地将噪声特征向量的平均值与噪声水平真值计算损失
            noise_feature = pred['noise_feature']  # [B, noise_feat_dim]
            noise_pred = noise_feature.mean(dim=1)  # [B]
            noise_loss = self.mse_loss(noise_pred, noise_level_gt)

        # 总损失
        total_loss = (angle_loss +
                      self.alpha * dolp_loss +
                      self.beta * consistency_loss +
                      self.gamma * noise_loss)

        return {
            'total': total_loss,
            'angle': angle_loss,
            'dolp': dolp_loss,
            'consistency': consistency_loss,
            'noise': noise_loss
        }


#===============================训练策略================================
# 数据加载策略
class PolarDataset(torch.utils.data.Dataset):
    def __init__(self, noisy_images, clean_images):
        """
        noisy_images: 形状 [N, 4, H, W] 包含四个噪声角度
        clean_images: 形状 [N, 5, H, W] 包含四个干净角度 + DOLP
        """
        self.noisy = noisy_images
        self.clean = clean_images

    def __getitem__(self, idx):
        noisy = self.noisy[idx]  # [4, H, W]
        clean = self.clean[idx]  # [5, H, W]

        return {
            'noisy': noisy,  # I0, I45, I90, I135
            'I0_gt': clean[0:1],
            'I45_gt': clean[1:2],
            'I90_gt': clean[2:3],
            'I135_gt': clean[3:4],
            'dolp_gt': clean[4:5]
        }


# 训练循环示例
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        # 将数据移到设备
        noisy = batch['noisy'].to(device)
        targets = {k: v.to(device) for k, v in batch.items() if k != 'noisy'}

        # 前向传播
        outputs = model(noisy[:, 0:1], noisy[:, 1:2],
                        noisy[:, 2:3], noisy[:, 3:4])

        # 计算损失
        loss_dict = criterion(outputs, targets)
        loss = loss_dict['total']

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
