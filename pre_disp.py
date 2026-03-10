# 导入必要的库，保持与原测试文件一致的依赖
import argparse
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import time
import numpy as np
from torch.autograd import Variable
from datasets import __datasets__  # 假设已实现SCARED数据集的加载类
from models import __models__, model_loss_test  # 复用模型和测试损失函数
from utils import *  # 复用工具函数（如评估指标、数据转换等）
from torch.utils.data import DataLoader
import cv2
import tifffile

# 启用cudnn加速
cudnn.benchmark = True
# 设置使用的GPU设备（根据实际情况调整）
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # SCARED数据集较小，可适当减少GPU数量


def save_disparity(disp, filename, output_dir='./pred_disparities'):
    """
    保存预测的视差图

    参数:
        disp: 视差图 numpy数组 [H, W]
        filename: 原始左图文件名（用于生成输出文件名）
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 从文件名生成输出路径
    base_name = os.path.basename(filename)
    name_without_ext = os.path.splitext(base_name)[0]

    # 保存为TIFF格式（保持精度）
    tiff_path = os.path.join(output_dir, f"{name_without_ext}_disp.tiff")
    tifffile.imwrite(tiff_path, disp.astype(np.float32))

    # 同时保存为PNG格式用于可视化
    png_path = os.path.join(output_dir, f"{name_without_ext}_disp.png")

    # 归一化到0-255范围用于可视化
    disp_vis = np.copy(disp)
    # 过滤无效值
    valid_mask = (disp_vis > 0) & (disp_vis < np.inf) & (~np.isnan(disp_vis))
    if np.any(valid_mask):
        valid_disp = disp_vis[valid_mask]
        vmin, vmax = np.percentile(valid_disp, [5, 95])  # 使用5%和95%分位数避免异常值
        disp_vis = np.clip(disp_vis, vmin, vmax)
        disp_vis = (disp_vis - vmin) / (vmax - vmin + 1e-8) * 255
    else:
        disp_vis = np.zeros_like(disp_vis)

    disp_vis = np.clip(disp_vis, 0, 255).astype(np.uint8)
    # 应用颜色映射
    disp_colored = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
    cv2.imwrite(png_path, disp_colored)

    print(f"Saved disparity map: {tiff_path}")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='测试SCARED数据集的立体匹配模型')
    # 模型选择（与原文件一致，假设模型兼容SCARED数据集）
    parser.add_argument('--model', default='acvnet', help='选择模型结构', choices=__models__.keys())
    # SCARED数据集视差范围较小，通常最大视差设为128（根据实际数据调整）
    parser.add_argument('--maxdisp', type=int, default=128, help='最大视差值（SCARED建议128）')
    # 数据集名称改为scared（需确保__datasets__中存在该键）
    parser.add_argument('--dataset', default='scared', help='数据集名称', choices=__datasets__.keys())
    # SCARED数据集路径（用户需替换为实际路径）
    parser.add_argument('--datapath', default="C:/Users/12700/Desktop/All_datasets/SCARED", help='SCARED数据集路径')
    # SCARED测试集列表文件（需用户提前准备，格式与原项目一致）
    parser.add_argument('--testlist', default='./filenames/scared_test.txt', help='测试集文件列表')
    # 测试批次大小（SCARED图像分辨率可能较高，适当减小批次）
    parser.add_argument('--test_batch_size', type=int, default=2, help='测试批次大小')
    # 预训练模型权重路径（需替换为适配SCARED的模型或通用模型）
    parser.add_argument('--loadckpt', default='./logs/scared/checkpoint_000059.ckpt', help='模型权重路径')
    # 新增：预测视差图保存路径
    parser.add_argument('--output_dir', default='./data/scared/pred_disparities', help='预测视差图保存路径')

    args = parser.parse_args()

    # 加载SCARED数据集
    # 假设__datasets__['scared']已实现，继承自基础立体数据集类
    StereoDataset = __datasets__[args.dataset]
    # 初始化测试数据集（第三个参数为False表示测试模式，不加载增强数据）
    test_dataset = StereoDataset(args.datapath, args.testlist, False)
    # 创建数据加载器：不打乱数据，根据GPU数量设置worker
    TestImgLoader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=8,  # SCARED数据量较小，减少worker数量
        drop_last=False  # 不丢弃最后一个不完整批次
    )

    # 初始化模型
    # 第三个参数为False表示测试模式（不启用dropout等训练特有的层）
    model = __models__[args.model](args.maxdisp, False, False)
    # 多GPU并行计算（根据CUDA_VISIBLE_DEVICES设置）
    model = nn.DataParallel(model)
    # 将模型移至GPU
    model.cuda()

    # 加载预训练权重
    print(f"加载模型权重：{args.loadckpt}")
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])  # 假设权重文件结构与原项目一致

    # 执行测试并生成视差图
    test(model, TestImgLoader, args)


def test(model, test_loader, args):
    """
    测试函数：遍历测试集，计算平均评估指标，并保存预测视差图
    Args:
        model: 训练好的立体匹配模型
        test_loader: 测试集数据加载器
        args: 命令行参数
    """
    # 用于累积测试过程中的评估指标
    avg_test_scalars = AverageMeterDict()

    for batch_idx, sample in enumerate(test_loader):
        start_time = time.time()
        # 测试单个批次并获取损失、评估指标和预测视差
        loss, scalar_outputs, pred_disparities = test_sample(model, sample, args)

        # 保存预测视差图
        batch_size = pred_disparities[0].shape[0] if isinstance(pred_disparities[0], torch.Tensor) else len(
            pred_disparities[0])

        for i in range(batch_size):
            # 获取第i个样本的预测视差（取最后一个尺度的预测，通常是最精确的）
            if isinstance(pred_disparities[-1], torch.Tensor):
                pred_disp = pred_disparities[-1][i].cpu().numpy()  # [H, W]
            else:
                pred_disp = pred_disparities[-1][i]  # 已经是numpy数组

            # 获取对应的左图文件名
            left_filename = sample['left_filename'][i] if 'left_filename' in sample else f'batch_{batch_idx}_sample_{i}'

            # 保存视差图
            save_disparity(pred_disp, left_filename, args.output_dir)

        # 更新平均指标
        avg_test_scalars.update(scalar_outputs)
        # 释放当前批次的指标内存
        del scalar_outputs
        # 打印当前批次信息
        print(f'迭代 {batch_idx}/{len(test_loader)}, 测试损失 = {loss:.3f}, 耗时 = {time.time() - start_time:.3f}s')

    # 计算所有批次的平均指标
    avg_test_scalars = avg_test_scalars.mean()
    print("测试集平均指标：", avg_test_scalars)


@make_nograd_func  # 工具函数装饰器：禁用梯度计算，加速测试
def test_sample(model, sample, args):
    """
    测试单个样本（批次）
    Args:
        model: 立体匹配模型
        sample: 包含左右图和真实视差的字典
        args: 命令行参数
    Returns:
        loss: 测试损失值
        scalar_outputs: 包含各项评估指标的字典
        pred_disparities: 预测的视差图列表
    """
    # 模型设为评估模式（关闭BatchNorm和Dropout）
    model.eval()

    # 从样本中提取数据并移至GPU
    imgL = sample['left'].cuda()  # 左图 [B, 3, H, W]
    imgR = sample['right'].cuda()  # 右图 [B, 3, H, W]
    disp_gt = sample['disparity'].cuda()  # 真实视差 [B, 1, H, W]

    # 生成有效视差掩码：仅计算视差在0~maxdisp范围内的区域
    # SCARED数据集可能存在无效视差（如负数或超出范围），需要过滤
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)

    # 模型预测视差（可能返回多尺度结果，如粗到精的预测）
    disp_ests = model(imgL, imgR)

    # 计算测试损失（根据模型设计，可能是多尺度损失）
    loss = model_loss_test(disp_ests, disp_gt, mask)

    # 计算各项评估指标（复用原工具函数）
    scalar_outputs = {"loss": loss}
    # EPE：端点误差（平均像素误差）
    scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    # D1：误差>3px或>5%的像素比例
    scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    # 不同阈值下的误差比例（1px、2px、3px）
    scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
    scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
    scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]

    # 将预测视差转换为numpy数组用于保存
    pred_disparities_np = []
    for disp_est in disp_ests:
        if isinstance(disp_est, torch.Tensor):
            pred_disparities_np.append(disp_est.detach().cpu().numpy())
        else:
            pred_disparities_np.append(disp_est)

    # 将张量转换为浮点数并返回
    return tensor2float(loss), tensor2float(scalar_outputs), pred_disparities_np


if __name__ == '__main__':
    main()