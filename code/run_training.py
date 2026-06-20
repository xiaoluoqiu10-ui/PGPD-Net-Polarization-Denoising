#!/usr/bin/env python3
"""
启动训练脚本
"""

import subprocess
import sys
import os


def install_requirements():
    """安装必需的包"""
    requirements = [
        'torch',
        'torchvision',
        'numpy',
        'opencv-python',
        'matplotlib',
        'tqdm',
        'scikit-image',
        'Pillow',
    ]

    for package in requirements:
        try:
            __import__(package.replace('-', '_').split('>=')[0])
        except ImportError:
            print(f"安装 {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])


def check_dependencies():
    """检查依赖"""
    print("检查依赖...")
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")

        import cv2
        print(f"✓ OpenCV {cv2.__version__}")

        import numpy as np
        print(f"✓ NumPy {np.__version__}")

        return True
    except ImportError as e:
        print(f"✗ 缺少依赖: {e}")
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("偏振图像去噪训练启动脚本")
    print("=" * 60)

    # 检查依赖
    # if not check_dependencies():
    #     print("\n尝试安装依赖...")
    #     install_requirements()
    #     if not check_dependencies():
    #         print("依赖安装失败，请手动安装所需的包")
    #         return

    # 检查数据集路径
    dataset_path = r'G:\zzz\2\new\data2.1\dataset_info.json'
    if not os.path.exists(dataset_path):
        print(f"\n✗ 数据集文件不存在: {dataset_path}")
        print("请检查路径是否正确")
        return

    print(f"\n✓ 数据集文件存在: {dataset_path}")

    # 创建必要的目录
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # 询问训练参数
    print("\n训练参数设置:")
    print("1. 快速训练（10个epoch，用于测试）")
    print("2. 标准训练（100个epoch）")
    print("3. 自定义参数")

    choice = input("\n请选择训练模式 (1/2/3): ").strip()

    if choice == '1':
        # 快速测试训练
        print("\n开始快速测试训练...")
        import train
        # 修改配置为快速训练
        train.config = train.config.copy()
        train.config['num_epochs'] = 10
        train.config['batch_size'] = 1  # 更小的batch_size加快训练
        train.config['print_freq'] = 5
        train.config['save_freq'] = 2
        train.main()

    elif choice == '2':
        # 标准训练
        print("\n开始标准训练...")
        import train
        train.main()

    elif choice == '3':
        # 自定义参数
        print("\n自定义训练参数:")

        # 这里可以添加更多的自定义选项
        import train
        train.main()

    else:
        print("无效选择，退出")
        return

    print("\n训练完成！")
    print("可以在 'checkpoints' 目录中找到训练结果")


if __name__ == "__main__":
    main()