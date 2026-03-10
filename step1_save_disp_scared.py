# 从__future__导入print_function和division，确保Python2/3兼容性
from __future__ import print_function, division
import argparse  # 用于解析命令行参数
import os  # 用于文件路径和目录操作
import torch  # PyTorch深度学习框架核心库
import torch.nn as nn  # PyTorch神经网络模块
import torch.nn.parallel  # 用于多GPU并行计算
import torch.backends.cudnn as cudnn  # 优化CUDA卷积计算
import torch.utils.data  # 数据加载和处理工具
from torch.autograd import Variable  # 自动求导变量（兼容旧代码）
import torch.nn.functional as F  # 神经网络函数库
# import numpy as np  # 数值计算和数组操作
import time  # 计时工具
from tensorboardX import SummaryWriter  # TensorBoard日志（预留）
from datasets import __datasets__  # 自定义数据集字典（包含SCARed数据集类）
from models import __models__  # 自定义模型字典（包含立体匹配模型）
from utils import *  # 自定义工具函数（如tensor2numpy、make_nograd_func等）
from torch.utils.data import DataLoader  # 批量数据加载器
import cv2  # OpenCV库，用于图像读写和可视化
import tifffile  # 用于保存TIFF格式的原始视差数据

# 设置使用的CUDA设备为第0块GPU（单GPU运行配置）
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 创建命令行参数解析器，描述为"立体匹配模型生成SCARed数据集视差图"
parser = argparse.ArgumentParser(description='Generate disparity maps for SCARed dataset using stereo matching model')
# 模型选择参数：默认使用"acvnet"，可选值为__models__字典中的模型名
parser.add_argument('--model', default='acvnet', help='select a model structure', choices=__models__.keys())
# 最大视差参数：根据SCARed数据集特点设置为192（可根据实际场景调整）
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
# 数据集选择参数：指定为SCARed数据集（需确保__datasets__中存在'scared'键）
parser.add_argument('--dataset', default='scared', help='dataset name', choices=__datasets__.keys())
# 数据路径参数：SCARed数据集的存储根路径（需用户根据实际路径修改）
parser.add_argument('--datapath', default="C:/Users/12700/Desktop/All_datasets/SCARED", help='SCARed dataset path')
# 测试列表参数：SCARed测试集样本路径列表文件（记录需处理的图像对路径）
parser.add_argument('--testlist', default='./filenames/scared_test.txt', help='SCARed testing list file')
# 加载模型参数：预训练模型的checkpoint路径（需使用适配SCARed的模型权重）
parser.add_argument('--loadckpt', default='./logs/scared/checkpoint_000059.ckpt', help='load weights from checkpoint')
# 解析命令行参数，得到参数对象args
args = parser.parse_args()

# 初始化SCARed数据集和数据加载器`
# 从自定义数据集字典中获取SCARed数据集类
StereoDataset = __datasets__[args.dataset]
# 创建测试数据集实例：传入数据路径、测试列表、训练模式（False表示测试）
test_dataset = StereoDataset(args.datapath, args.testlist, False)
# 创建数据加载器：批量大小1（单对图像处理），不打乱数据，4个工作进程
TestImgLoader = DataLoader(test_dataset, 1, shuffle=False, num_workers=4, drop_last=False)

# 初始化模型并配置GPU
# 从模型字典中加载指定模型，传入最大视差和测试模式参数（后两个False表示测试）
model = __models__[args.model](args.maxdisp, False, False)
# 使用DataParallel包装模型，支持多GPU（单GPU环境下仍兼容）
model = nn.DataParallel(model)
# 将模型移动到GPU上运行
model.cuda()

# 加载预训练模型权重
print("Loading model from {}".format(args.loadckpt))  # 打印加载的模型路径
state_dict = torch.load(args.loadckpt)  # 加载checkpoint文件
model.load_state_dict(state_dict['model'])  # 将权重加载到模型中

# 视差图保存目录：SCARed数据集视差结果存储路径（需用户根据实际路径修改）
save_dir = './data/scared/disp/' # 彩色视差数据保存路径
save_dir_raw = './data/scared/disp_raw/' #原始视差数据保存路径
save_dir_gray = './data/scared/disp_gray/'  # 新增：灰度视差图保存路径


def test():
    """测试主函数：遍历SCARed测试集，生成并保存视差图"""
    # 判断保存目录是否存在，不存在则创建（包括多级目录）
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)  # exist_ok=True避免目录已存在时报错

    if not os.path.exists(save_dir_raw):
        os.makedirs(save_dir_raw, exist_ok=True)

    # 新增：创建灰度视差图保存目录
    if not os.path.exists(save_dir_gray):
        os.makedirs(save_dir_gray, exist_ok=True)

    # 遍历测试数据加载器中的每个样本
    for batch_idx, sample in enumerate(TestImgLoader):
        torch.cuda.synchronize()  # 等待GPU操作完成，确保计时准确
        start_time = time.time()  # 记录当前样本处理开始时间

        # 对样本进行测试，得到视差图的numpy数组（通过tensor2numpy转换）
        disp_est_np = tensor2numpy(test_sample(sample))

        torch.cuda.synchronize()  # 等待GPU操作完成
        # 打印当前迭代进度和处理耗时
        print('Iter {}/{}, time = {:.3f}s'.format(
            batch_idx, len(TestImgLoader), time.time() - start_time
        ))

        # 获取左图文件名列表（用于生成视差图保存路径）
        left_filenames = sample["left_filename"]

        # 遍历当前批次的视差图和对应文件名，执行保存操作
        for disp_est, fn in zip(disp_est_np, left_filenames):
            # 确保视差图为二维数组（高度×宽度）
            assert len(disp_est.shape) == 2
            # 转换视差图数据类型为float64
            disp_est = np.array(disp_est, dtype=np.float64)

            # 构造保存路径：根据SCARed数据集的文件结构提取关键路径部分
            # 注：split('/')的索引需根据实际文件路径调整，确保生成唯一文件名
            base_name = fn.split('/')[-3] + '_' + fn.split('/')[-1]
            fn_color = os.path.join(save_dir, base_name)
            fn_raw = os.path.join(save_dir_raw, os.path.splitext(base_name)[0] + '.tiff')
            fn_gray = os.path.join(save_dir_gray, base_name)  # 新增：灰度视差图路径
            print("Saving disparity to", fn, "| Shape:", disp_est.shape)
            print("Saving raw disparity to", fn_raw)
            print("Saving gray disparity to", fn_gray)  # 新增：打印灰度图保存路径

            # 视差图预处理：乘以256并转为uint16（保留精度同时减少存储）
            disp_est_uint = np.round(disp_est * 256).astype(np.uint16)
            # 生成可视化彩色视差图：缩放至0-255范围后应用JET颜色映射
            cv2.imwrite(
                fn_color,
                cv2.applyColorMap(
                    cv2.convertScaleAbs(disp_est_uint, alpha=0.01),  # 缩放系数根据视差范围调整
                    cv2.COLORMAP_JET
                )
            )
            # 保存原始视差数据为TIFF格式（使用tifffile保留完整浮点精度）
            tifffile.imwrite(fn_raw, disp_est)

            # 新增：保存灰度视差图
            # 归一化视差值到0-255范围（线性拉伸）
            disp_min = disp_est.min()
            disp_max = disp_est.max()
            if disp_max > disp_min:  # 避免除以零
                disp_gray = ((disp_est - disp_min) / (disp_max - disp_min) * 255).astype(np.uint8)
            else:
                disp_gray = np.zeros_like(disp_est, dtype=np.uint8)
            cv2.imwrite(fn_gray, disp_gray)  # 保存单通道灰度图

# 装饰器禁用梯度计算（测试阶段无需反向传播，节省内存并加速）
@make_nograd_func
def test_sample(sample):
    """测试单个样本：输入左右图像，输出模型预测的视差图"""
    model.eval()  # 模型切换至评估模式（禁用dropout等训练特有层）
    # 将左右图像数据移动到GPU，输入模型得到视差估计结果（取最后一个尺度的输出）
    disp_ests = model(sample['left'].cuda(), sample['right'].cuda())
    return disp_ests[-1]  # 返回最终视差图

# 主程序入口：执行测试流程
if __name__ == '__main__':
    test()