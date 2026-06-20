import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import json
import os


class PolarDenoisingDataset(Dataset):
    """偏振图像去噪数据集"""

    def __init__(self, dataset_info_path, transform=None):
        """
        dataset_info_path: dataset_info.json的路径
        transform: 数据增强变换
        """
        with open(dataset_info_path, 'r', encoding='utf-8') as f:
            self.dataset_info = json.load(f)

        self.transform = transform
        self.samples = []

        # 创建样本列表：每个噪声等级和场景的组合都是一个样本
        for scene_info in self.dataset_info:
            scene = scene_info['scene']

            # 遍历所有噪声等级（500, 1000, 2000, 4000）
            for noise_level, noise_files in scene_info['noise_images'].items():
                self.samples.append({
                    'scene': scene,
                    'noise_level': noise_level,  # 噪声等级
                    'noise_paths': noise_files,  # 噪声图像路径
                    'gt_paths': scene_info['gt_images']  # 对应的真值图像路径
                })

        print(f"数据集加载完成，共 {len(self.samples)} 个样本")
        print(f"场景数: {len(self.dataset_info)}")
        print(f"噪声等级: 500, 1000, 2000, 4000")

    def __len__(self):
        return len(self.samples)

    def load_image(self, path):
        """
        加载8位BMP图像，转换为float32并归一化到[0.0, 1.0]
        返回形状为 (H, W) 的numpy数组
        """
        # 尝试读取图像
        if not os.path.exists(path):
            raise FileNotFoundError(f"图像文件不存在: {path}")

        # 读取灰度图像
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"无法加载图像，可能是格式错误: {path}")

        # 检查图像尺寸
        if img.shape != (1024, 1224):  # 你的图像尺寸
            print(f"警告: 图像 {path} 尺寸为 {img.shape}，期望 (1024, 1224)")

        # 归一化到[0, 1]
        img = img.astype(np.float32) / 255.0

        # 确保没有无效值
        img = np.clip(img, 0.0, 1.0)

        return img

    def __getitem__(self, idx):
        sample = self.samples[idx]

        try:
            # ========== 加载噪声图像（网络输入） ==========
            noise_images = []
            for angle in ['I0', 'I45', 'I90', 'I135']:
                img_path = sample['noise_paths'][angle]
                img = self.load_image(img_path)
                noise_images.append(img)  # 形状 (H, W)

            # ========== 加载真值图像（训练目标） ==========
            # 四个角度的真值图像
            I0_gt = self.load_image(sample['gt_paths']['I0'])
            I45_gt = self.load_image(sample['gt_paths']['I45'])
            I90_gt = self.load_image(sample['gt_paths']['I90'])
            I135_gt = self.load_image(sample['gt_paths']['I135'])

            # DOLP真值
            dolp_gt = self.load_image(sample['gt_paths']['dolp'])

            # ========== 转换为tensor ==========
            # 噪声图像组合成一个tensor: [4, H, W]
            noise_tensor = torch.from_numpy(np.stack(noise_images, axis=0)).float()

            # 真值图像
            I0_gt_tensor = torch.from_numpy(I0_gt).float().unsqueeze(0)  # [1, H, W]
            I45_gt_tensor = torch.from_numpy(I45_gt).float().unsqueeze(0)
            I90_gt_tensor = torch.from_numpy(I90_gt).float().unsqueeze(0)
            I135_gt_tensor = torch.from_numpy(I135_gt).float().unsqueeze(0)
            dolp_gt_tensor = torch.from_numpy(dolp_gt).float().unsqueeze(0)

            # 数据增强（如果有）
            if self.transform:
                # 这里可以添加随机裁剪、旋转等数据增强
                pass

            return {
                # 输入（噪声图像）
                'noisy_I0': noise_tensor[0:1],  # [1, H, W]
                'noisy_I45': noise_tensor[1:2],
                'noisy_I90': noise_tensor[2:3],
                'noisy_I135': noise_tensor[3:4],
                'noisy_full': noise_tensor,  # [4, H, W] 完整噪声图像

                # 目标（干净图像）
                'I0_gt': I0_gt_tensor,
                'I45_gt': I45_gt_tensor,
                'I90_gt': I90_gt_tensor,
                'I135_gt': I135_gt_tensor,
                'dolp_gt': dolp_gt_tensor,

                # 元数据
                'scene': sample['scene'],
                'noise_level': int(sample['noise_level']),
                'scene_idx': int(sample['scene']) - 1  # 0-based索引
            }

        except Exception as e:
            print(f"加载样本 {idx} 时出错 (场景{sample['scene']}, 噪声等级{sample['noise_level']}): {e}")
            print(f"噪声图像路径: {sample['noise_paths']}")
            # 返回第一个样本作为替代
            if idx > 0:
                return self.__getitem__(0)
            else:
                # 创建虚拟数据避免崩溃
                H, W = 1024, 1224
                dummy_img = torch.zeros((1, H, W))
                return {
                    'noisy_I0': dummy_img,
                    'noisy_I45': dummy_img,
                    'noisy_I90': dummy_img,
                    'noisy_I135': dummy_img,
                    'noisy_full': torch.zeros((4, H, W)),
                    'I0_gt': dummy_img,
                    'I45_gt': dummy_img,
                    'I90_gt': dummy_img,
                    'I135_gt': dummy_img,
                    'dolp_gt': dummy_img,
                    'scene': '1',
                    'noise_level': 500,
                    'scene_idx': 0
                }


def create_train_val_loaders(dataset_info_path, batch_size=4, train_ratio=0.8,
                             random_seed=42, num_workers=4, shuffle_train=True):
    """创建训练和验证数据加载器"""

    print(f"创建数据加载器，数据集路径: {dataset_info_path}")

    # 创建完整数据集
    full_dataset = PolarDenoisingDataset(dataset_info_path)

    # 计算分割大小
    train_size = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size

    print(f"总样本数: {len(full_dataset)}")
    print(f"训练集大小: {train_size}")
    print(f"验证集大小: {val_size}")

    # 固定随机种子以确保可重复性
    generator = torch.Generator().manual_seed(random_seed)

    # 分割数据集
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False
    )

    # 打印数据集统计信息
    print("\n数据集统计:")
    print(f"  场景范围: 1-21")
    print(f"  噪声等级: 500, 1000, 2000, 4000")
    print(f"  图像尺寸: 1024×1224")
    print(f"  训练批次大小: {batch_size}")
    print(f"  训练迭代次数/epoch: {len(train_loader)}")

    return train_loader, val_loader, full_dataset


# 测试函数
def test_dataset():
    """测试数据集加载"""
    # 使用正确的路径
    dataset_path = r'G:\zzz\2\new\data2.1\dataset_info.json'

    print("测试数据集加载...")
    print(f"数据集路径: {dataset_path}")

    try:
        dataset = PolarDenoisingDataset(dataset_path)

        # 获取第一个样本
        sample = dataset[0]

        print("\n样本信息:")
        print(f"  噪声图像形状: {sample['noisy_full'].shape}")
        print(f"  I0噪声形状: {sample['noisy_I0'].shape}")
        print(f"  I0真值形状: {sample['I0_gt'].shape}")
        print(f"  DOLP真值形状: {sample['dolp_gt'].shape}")
        print(f"  场景: {sample['scene']}")
        print(f"  噪声等级: {sample['noise_level']}")
        print(f"  场景索引: {sample['scene_idx']}")

        # 检查数据范围
        print("\n数据范围检查:")
        print(f"  噪声图像范围: [{sample['noisy_full'].min():.3f}, {sample['noisy_full'].max():.3f}]")
        print(f"  真值图像范围: [{sample['I0_gt'].min():.3f}, {sample['I0_gt'].max():.3f}]")

        # 测试数据加载器
        print("\n测试数据加载器...")
        train_loader, val_loader, _ = create_train_val_loaders(
            dataset_path,
            batch_size=2
        )

        # 查看一个batch
        for batch in train_loader:
            print(f"\nBatch信息:")
            print(f"  噪声图像形状: {batch['noisy_full'].shape}")
            print(f"  I0真值形状: {batch['I0_gt'].shape}")
            print(f"  场景列表: {batch['scene']}")
            print(f"  噪声等级列表: {batch['noise_level']}")

            # 检查是否所有数据都在[0, 1]范围内
            print(f"  数据范围检查 - 噪声: [{batch['noisy_full'].min():.3f}, {batch['noisy_full'].max():.3f}]")
            print(f"  数据范围检查 - 真值: [{batch['I0_gt'].min():.3f}, {batch['I0_gt'].max():.3f}]")
            break

    except Exception as e:
        print(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_dataset()