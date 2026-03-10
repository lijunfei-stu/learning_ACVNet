# 文件：main_origin.py
# 作用：训练和测试 ACVNet 立体匹配模型的主脚本
# 使用 PyTorch 框架，支持多 GPU 训练，包含数据加载、模型定义、训练循环、测试循环和日志记录等功能。

# 导入未来的 print 函数和除法功能（Python 2/3 兼容，但在 Python 3 中不需要）
# from __future__ import print_function, division

import argparse  # 用于解析命令行参数
import os        # 提供与操作系统交互的功能，如文件路径、创建目录等
import torch     # PyTorch 主库
import torch.nn as nn  # 神经网络模块
import torch.nn.parallel  # 多 GPU 并行训练支持
import torch.backends.cudnn as cudnn  # cuDNN 优化库，加速卷积运算
import torch.optim as optim  # 优化器，如 Adam、SGD
import torch.utils.data  # 数据加载工具
from torch.autograd import Variable  # 自动求导变量（在 PyTorch 0.4.0 之后 Tensor 和 Variable 合并，此处可能为兼容旧代码）
import torchvision.utils as vutils  # 图像处理工具，如保存图像网格
import torch.nn.functional as F  # 包含常用的神经网络函数（如激活函数、损失函数等）
import numpy as np  # 数值计算库
import time  # 时间模块，用于计时
from tensorboardX import SummaryWriter  # 用于将训练过程记录到 TensorBoard，可视化损失、图像等
from datasets import __datasets__  # 从自定义数据集模块中导入数据集字典
from models import __models__, model_loss_train_attn_only, model_loss_train_freeze_attn, model_loss_train, \
    model_loss_test  # 导入模型字典和不同的损失函数
from utils import *  # 导入自定义工具函数，如学习率调整、指标计算、图像保存等
from torch.utils.data import DataLoader  # 数据加载器，用于批量加载数据
import gc  # 垃圾回收模块，用于手动释放内存
# from apex import amp  # 混合精度训练工具（被注释掉，未使用）
import cv2  # OpenCV 库，用于图像处理

# 设置 cuDNN 的 benchmark 模式为 True，让 cuDNN 自动选择最优卷积算法，加速训练（通常当输入尺寸固定时有效）
cudnn.benchmark = True
# 设置环境变量 CUDA_VISIBLE_DEVICES，指定使用的 GPU 编号（这里使用 0,1,2,3 四块 GPU）
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

# 创建命令行参数解析器
parser = argparse.ArgumentParser(
    description='Attention Concatenation Volume for Accurate and Efficient Stereo Matching (ACVNet)')
# 添加各种命令行参数，用户可以在运行脚本时指定这些参数的值
parser.add_argument('--model', default='acvnet', help='select a model structure', choices=__models__.keys())
# 模型名称，默认为 acvnet，可选值来自 __models__ 字典的键
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
# 最大视差值，用于限制视差搜索范围
parser.add_argument('--dataset', default='sceneflow', help='dataset name', choices=__datasets__.keys())
# 数据集名称，默认为 sceneflow，可选值来自 __datasets__ 字典的键
parser.add_argument('--datapath', default="/data/sceneflow/", help='data path')
# 数据集路径
parser.add_argument('--trainlist', default='./filenames/sceneflow_train.txt', help='training list')
# 训练集文件列表的路径，文件中每行指定一对左右图像和对应的视差图
parser.add_argument('--testlist', default='./filenames/sceneflow_test.txt', help='testing list')
# 测试集文件列表的路径
parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
# 初始学习率
parser.add_argument('--batch_size', type=int, default=20, help='training batch size')
# 训练时的批量大小（每个 GPU 上的样本数，实际总批量大小 = batch_size * GPU 数量）
parser.add_argument('--test_batch_size', type=int, default=16, help='testing batch size')
# 测试时的批量大小
parser.add_argument('--epochs', type=int, default=64, help='number of epochs to train')
# 训练的总轮数（一个 epoch 表示遍历一次整个训练集）
parser.add_argument('--lrepochs', default="20,32,40,48,56:2", type=str,
                    help='the epochs to decay lr: the downscale rate')
# 学习率衰减策略：格式如 "epoch1,epoch2,...:scale"，表示在指定的 epoch 将学习率除以 scale
parser.add_argument('--attention_weights_only', default=False, type=str, help='only train attention weights')
# 是否只训练注意力权重部分（字符串类型，实际应转换为布尔值）
parser.add_argument('--freeze_attention_weights', default=False, type=str, help='freeze attention weights parameters')
# 是否冻结注意力权重参数（不训练）
# parser.add_argument('--lrepochs',default="300,500:2", type=str,  help='the epochs to decay lr: the downscale rate')
# 另一条学习率策略（被注释掉）
parser.add_argument('--logdir', default='', help='the directory to save logs and checkpoints')
# 日志和模型保存的目录
parser.add_argument('--loadckpt', default='./pretrained_model/pretrained_model_sceneflow.ckpt',
                    help='load the weights from a specific checkpoint')
# 加载预训练模型检查点的路径
parser.add_argument('--resume', action='store_true', help='continue training the model')
# 是否恢复训练（从上次保存的检查点继续）
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
# 随机种子，用于结果可重复
parser.add_argument('--summary_freq', type=int, default=20, help='the frequency of saving summary')
# 保存训练摘要（如损失、图像）的频率（每多少个 global step 保存一次）
parser.add_argument('--save_freq', type=int, default=1, help='the frequency of saving checkpoint')
# 保存模型检查点的频率（每多少个 epoch 保存一次）

# 解析命令行参数，将结果存入 args 对象
args = parser.parse_args()

# 设置 PyTorch 的随机种子，确保结果可复现
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# 创建日志目录，如果目录已存在则不会报错
os.makedirs(args.logdir, exist_ok=True)

# 创建 TensorBoard 的 SummaryWriter，用于记录训练过程中的标量、图像等信息
print("creating new summary file")
logger = SummaryWriter(args.logdir)

# 根据数据集名称获取对应的数据集类
StereoDataset = __datasets__[args.dataset]
# 创建训练集实例，参数：数据路径，训练文件列表，是否用于训练（True 表示需要数据增强等）
train_dataset = StereoDataset(args.datapath, args.trainlist, True)
# 创建测试集实例，参数：数据路径，测试文件列表，是否用于训练（False 表示不需要数据增强）
test_dataset = StereoDataset(args.datapath, args.testlist, False)

# 创建训练数据加载器：批量大小 args.batch_size，每个 epoch 打乱数据，使用 16 个子进程加载数据，丢弃最后一批可能不足 batch_size 的数据
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=16, drop_last=True)
# 创建测试数据加载器：批量大小 args.test_batch_size，不打乱，使用 16 个子进程，不丢弃最后一批
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=16, drop_last=False)

# 根据模型名称获取模型类，并实例化模型，传入最大视差和两个标志参数
model = __models__[args.model](args.maxdisp, args.attention_weights_only, args.freeze_attention_weights)
# 使用 DataParallel 包装模型，使模型能在多个 GPU 上并行训练
model = nn.DataParallel(model)
# 将模型移动到 GPU（默认使用所有可见 GPU）
model.cuda()

# 创建优化器，使用 Adam 优化算法，传入模型参数和学习率
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
# 另一条注释掉的优化器：SGD
# optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum=0.9)

# 初始化起始 epoch 为 0
start_epoch = 0

# 如果指定了 resume（恢复训练）
if args.resume:
    # 列出日志目录下所有以 .ckpt 结尾的文件
    all_saved_ckpts = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    # 按照文件名中的 epoch 数字排序（文件名格式如 checkpoint_000001.ckpt）
    all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # 使用最新的检查点文件
    loadckpt = os.path.join(args.logdir, all_saved_ckpts[-1])
    print("loading the lastest model in logdir: {}".format(loadckpt))
    # 加载检查点文件（包含模型状态、优化器状态和 epoch）
    state_dict = torch.load(loadckpt)
    # 加载模型参数
    model.load_state_dict(state_dict['model'])
    # 加载优化器参数
    optimizer.load_state_dict(state_dict['optimizer'])
    # 设置起始 epoch 为保存的 epoch + 1（从下一个 epoch 开始训练）
    start_epoch = state_dict['epoch'] + 1
# 如果指定了 loadckpt（加载预训练模型，但不恢复训练）
elif args.loadckpt:
    print("loading model {}".format(args.loadckpt))
    # 加载检查点文件
    state_dict = torch.load(args.loadckpt)
    # 获取当前模型的参数字典
    model_dict = model.state_dict()
    # 筛选出预训练模型中与当前模型键匹配的参数（只加载匹配的部分）
    pre_dict = {k: v for k, v in state_dict['model'].items() if k in model_dict}
    # 更新当前模型参数字典
    model_dict.update(pre_dict)
    # 加载更新后的参数
    model.load_state_dict(model_dict)

print("start at epoch {}".format(start_epoch))


def train():
    """主训练函数，执行多个 epoch 的训练和测试"""
    for epoch_idx in range(start_epoch, args.epochs):
        # 根据当前 epoch 调整学习率（调用自定义工具函数 adjust_learning_rate）
        adjust_learning_rate(optimizer, epoch_idx, args.lr, args.lrepochs)

        # 训练阶段
        for batch_idx, sample in enumerate(TrainImgLoader):
            # 计算全局 step（所有 epoch 中已处理的 batch 总数）
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            # 判断当前 step 是否需要保存摘要
            do_summary = global_step % args.summary_freq == 0
            # 训练一个 batch 的样本，返回损失、标量输出和图像输出
            loss, scalar_outputs, image_outputs = train_sample(sample, compute_metrics=do_summary)
            if do_summary:
                # 保存标量到 TensorBoard
                save_scalars(logger, 'train', scalar_outputs, global_step)
                # 保存图像到 TensorBoard
                save_images(logger, 'train', image_outputs, global_step)
            # 删除变量以释放内存
            del scalar_outputs, image_outputs
            # 打印训练信息
            print('Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, args.epochs,
                                                                                       batch_idx,
                                                                                       len(TrainImgLoader), loss,
                                                                                       time.time() - start_time))

        # 保存模型检查点
        if (epoch_idx + 1) % args.save_freq == 0:
            checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            # 保存检查点文件，文件名中包含 epoch 编号（6位数字）
            torch.save(checkpoint_data, "{}/checkpoint_{:0>6}.ckpt".format(args.logdir, epoch_idx))
        # 手动触发垃圾回收，释放内存
        gc.collect()

        # 每个 epoch 结束后进行测试（这里条件为 (epoch_idx) % 1 == 0，即每个 epoch 都测试）
        if (epoch_idx) % 1 == 0:
            # 创建一个平均计量器，用于记录测试过程中的标量指标的平均值
            avg_test_scalars = AverageMeterDict()
            for batch_idx, sample in enumerate(TestImgLoader):
                global_step = len(TestImgLoader) * epoch_idx + batch_idx
                start_time = time.time()
                do_summary = global_step % args.summary_freq == 0
                # 测试一个 batch
                loss, scalar_outputs, image_outputs = test_sample(sample, compute_metrics=do_summary)
                if do_summary:
                    save_scalars(logger, 'test', scalar_outputs, global_step)
                    save_images(logger, 'test', image_outputs, global_step)
                # 更新平均值
                avg_test_scalars.update(scalar_outputs)
                del scalar_outputs, image_outputs
                print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch_idx, args.epochs,
                                                                                         batch_idx,
                                                                                         len(TestImgLoader), loss,
                                                                                         time.time() - start_time))
            # 计算所有测试 batch 的平均标量
            avg_test_scalars = avg_test_scalars.mean()
            # 保存平均标量到 TensorBoard（使用全局 step = len(TrainImgLoader) * (epoch_idx + 1) 作为 x 轴）
            save_scalars(logger, 'fulltest', avg_test_scalars, len(TrainImgLoader) * (epoch_idx + 1))
            print("avg_test_scalars", avg_test_scalars)
            gc.collect()


# 训练一个 batch 的函数
def train_sample(sample, compute_metrics=False):
    model.train()  # 设置模型为训练模式（启用 dropout 等训练专用层）
    # 从 sample 字典中获取左图、右图和视差真值
    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    # 将数据移动到 GPU
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()
    # 清空优化器的梯度（避免累积）
    optimizer.zero_grad()
    # 前向传播：模型输入左右图像，输出视差估计值（可能是一个列表，包含不同阶段的输出）
    disp_ests = model(imgL, imgR)
    # 创建掩码，标记有效视差（在 0 到 maxdisp 之间），用于计算损失时忽略无效像素
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    # 根据命令行参数选择不同的损失函数
    if args.attention_weights_only:
        loss = model_loss_train_attn_only(disp_ests, disp_gt, mask)
    elif args.freeze_attention_weights:
        loss = model_loss_train_freeze_attn(disp_ests, disp_gt, mask)
    else:
        loss = model_loss_train(disp_ests, disp_gt, mask)
    # 准备标量输出（损失）
    scalar_outputs = {"loss": loss}
    # 准备图像输出（用于 TensorBoard 可视化）
    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}
    # 如果需要计算更多指标（compute_metrics=True）
    if compute_metrics:
        with torch.no_grad():  # 禁止梯度计算，节省内存和加速
            # 计算误差图（使用自定义函数 disp_error_image_func，可能将误差转换为彩色图）
            image_outputs["errormap"] = [disp_error_image_func.apply(disp_est, disp_gt) for disp_est in disp_ests]
            # 计算终点误差（EPE）
            scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
            # 计算 D1 指标（视差误差 > 3 或相对误差 > 5% 的像素比例）
            scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
            # 计算阈值误差（误差大于 1,2,3 像素的像素比例）
            scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
            scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
            scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]
    # 反向传播，计算梯度
    loss.backward()
    # 更新模型参数
    optimizer.step()
    # 将损失和标量输出中的张量转换为 Python 浮点数（便于打印和记录），图像输出保持不变
    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


# 测试一个 batch 的函数
@make_nograd_func  # 自定义装饰器，确保函数内不计算梯度（即使模型在 eval 模式，此装饰器可能进一步确保）
def test_sample(sample, compute_metrics=True):
    model.eval()  # 设置模型为评估模式（关闭 dropout 等）
    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    # 前向传播，得到视差估计
    disp_ests = model(imgL, imgR)
    # 复制真值以匹配估计的数量（模型可能输出多个尺度，这里用 6 个真值对应，但实际代码中 disp_gts 并未使用）
    disp_gts = [disp_gt, disp_gt, disp_gt, disp_gt, disp_gt, disp_gt]  # 此行变量未使用，可能是遗留代码
    # 计算测试损失（使用专门的测试损失函数）
    loss = model_loss_test(disp_ests, disp_gt, mask)
    scalar_outputs = {"loss": loss}
    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}
    # 计算误差图和各项指标（测试时默认计算）
    image_outputs["errormap"] = [disp_error_image_func.apply(disp_est, disp_gt) for disp_est in disp_ests]
    scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
    scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
    scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


# 如果该脚本作为主程序运行（而不是被导入），则调用 train 函数开始训练
if __name__ == '__main__':
    train()