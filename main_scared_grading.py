# from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import __datasets__  # 假设已实现SCARED数据集的加载类
from models import __models__, model_loss_train_attn_only, model_loss_train_freeze_attn, model_loss_train, \
    model_loss_test
from utils import *
from torch.utils.data import DataLoader
import gc
import cv2

cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 根据可用GPU调整

parser = argparse.ArgumentParser(description='ACVNet for SCARED Dataset Training')
# 模型相关参数
parser.add_argument('--model', default='acvnet', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192,help='maximum disparity (SCARED数据集视差范围可根据实际情况调整)')
# 数据集相关参数 - 适配SCARED数据集
parser.add_argument('--dataset', default='scared', help='dataset name', choices=__datasets__.keys())  # 修改为scared
parser.add_argument('--datapath', default="C:/Users/12700/Desktop/All_datasets/SCARED",help='SCARED数据集路径')  # 修改为SCARED路径
parser.add_argument('--trainlist', default='./filenames/scared_train.txt', help='SCARED训练列表文件')  # 修改为SCARED训练列表
parser.add_argument('--testlist', default='./filenames/scared_test.txt', help='SCARED测试列表文件')  # 修改为SCARED测试列表
# 训练相关参数 - 可根据SCARED数据集特点调整
parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
parser.add_argument('--batch_size', type=int, default=1,help='training batch size (SCARED图像尺寸可能较大，建议减小批次)')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--epochs', type=int, default=64, help='number of epochs to train (可根据SCARED数据集大小调整)')
parser.add_argument('--lrepochs', default="20,32,40,48,56:2", type=str, help='学习率衰减节点 (适配SCARED训练轮数)')
parser.add_argument('--attention_weights_only', default=False, type=bool, help='only train attention weights')
parser.add_argument('--freeze_attention_weights', default=False, type=bool, help='freeze attention weights parameters')
# 日志和checkpoint相关参数
parser.add_argument('--logdir', default='./logs/scared', help='SCARED训练日志和 checkpoint 保存根路径')
parser.add_argument('--loadckpt', default='', help='加载预训练模型路径 (可选)')
parser.add_argument('--resume', action='store_true', help='continue training the model')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
parser.add_argument('--summary_freq', type=int, default=20, help='summary保存频率')
parser.add_argument('--save_freq', type=int, default=5, help='checkpoint保存频率 (SCARED可适当提高保存间隔)')

# 解析参数并设置随机种子
args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# ********** 新增：根据训练模式生成独立保存目录 **********
# 避免不同训练阶段的权重文件相互覆盖
if args.attention_weights_only:
    args.logdir = f"{args.logdir}_attn_only"  # 仅训练注意力权重
elif args.freeze_attention_weights:
    args.logdir = f"{args.logdir}_freeze_attn"  # 冻结注意力权重训练
else:
    args.logdir = f"{args.logdir}_full"  # 联合训练所有模块
os.makedirs(args.logdir, exist_ok=True)

# 创建日志记录器
print(f"Creating new summary file for SCARED training (mode: {args.logdir.split('_')[-1]})")
logger = SummaryWriter(args.logdir)

# 数据集和数据加载器 - 适配SCARED
StereoDataset = __datasets__[args.dataset]
train_dataset = StereoDataset(args.datapath, args.trainlist, True)  # 第三个参数为True表示训练模式（可能包含数据增强）
test_dataset = StereoDataset(args.datapath, args.testlist, False)  # 测试模式
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=0,
                            drop_last=True)  # SCARED可能需要调整num_workers
print("TrainImgLoader initialized, total samples: ", len(TrainImgLoader.dataset))
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=0, drop_last=False)

# 模型和优化器
model = __models__[args.model](args.maxdisp, args.attention_weights_only, args.freeze_attention_weights)
model = nn.DataParallel(model)
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

# 加载模型参数
start_epoch = 0
# ********** 新增：最佳权重跟踪变量 **********
best_epe = float('inf')  # 初始最佳EPE设为无穷大
best_checkpoint_path = os.path.join(args.logdir, "best_checkpoint.ckpt")

if args.resume:
    all_saved_ckpts = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt") and not fn.startswith("best")]
    if all_saved_ckpts:
        all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        loadckpt = os.path.join(args.logdir, all_saved_ckpts[-1])
        print(f"Loading latest checkpoint from logdir: {loadckpt}")
        state_dict = torch.load(loadckpt)
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        start_epoch = state_dict['epoch'] + 1

        # 恢复最佳EPE记录（如果存在）
        if os.path.exists(best_checkpoint_path):
            best_state = torch.load(best_checkpoint_path)
            best_epe = best_state.get('best_epe', float('inf'))
            print(f"Resumed best EPE: {best_epe:.4f}")
elif args.loadckpt:
    print(f"Loading specified checkpoint: {args.loadckpt}")
    state_dict = torch.load(args.loadckpt)
    model_dict = model.state_dict()
    pre_dict = {k: v for k, v in state_dict['model'].items() if k in model_dict}
    model_dict.update(pre_dict)
    model.load_state_dict(model_dict)
print(f"Start training from epoch {start_epoch}")


def train():
    global best_epe  # 声明使用全局变量跟踪最佳EPE
    print("Entering training loop, preparing first batch...")
    for epoch_idx in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch_idx, args.lr, args.lrepochs)

        # 训练过程
        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = train_sample(sample, compute_metrics=do_summary)
            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)
                save_images(logger, 'train', image_outputs, global_step)
            del scalar_outputs, image_outputs
            print(
                f'Epoch {epoch_idx}/{args.epochs}, Iter {batch_idx}/{len(TrainImgLoader)}, train loss = {loss:.3f}, time = {time.time() - start_time:.3f}')

        # 按频率保存checkpoint（原有功能保留）
        if (epoch_idx + 1) % args.save_freq == 0:
            checkpoint_data = {
                'epoch': epoch_idx,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'current_epe': None  # 后续测试后补充
            }
            freq_ckpt_path = f"{args.logdir}/checkpoint_{epoch_idx:0>6}.ckpt"
            torch.save(checkpoint_data, freq_ckpt_path)
            print(f"Saved periodic checkpoint to {freq_ckpt_path}")

        gc.collect()

        # 每轮训练后进行测试
        if epoch_idx % 1 == 0:
            avg_test_scalars = AverageMeterDict()
            for batch_idx, sample in enumerate(TestImgLoader):
                global_step = len(TestImgLoader) * epoch_idx + batch_idx
                start_time = time.time()
                do_summary = global_step % args.summary_freq == 0
                loss, scalar_outputs, image_outputs = test_sample(sample, compute_metrics=do_summary)
                if do_summary:
                    save_scalars(logger, 'test', scalar_outputs, global_step)
                    save_images(logger, 'test', image_outputs, global_step)
                avg_test_scalars.update(scalar_outputs)
                del scalar_outputs, image_outputs
                print(
                    f'Epoch {epoch_idx}/{args.epochs}, Iter {batch_idx}/{len(TestImgLoader)}, test loss = {loss:.3f}, time = {time.time() - start_time:.3f}')

            avg_test_scalars = avg_test_scalars.mean()
            save_scalars(logger, 'fulltest', avg_test_scalars, len(TrainImgLoader) * (epoch_idx + 1))
            print(f"Average test metrics: {avg_test_scalars}")

            # ********** 新增：最佳权重判断与保存 **********
            # 根据训练模式选择对应的EPE指标（多输出时取最终预测的EPE）
            current_epe = avg_test_scalars['EPE'][-1] if isinstance(avg_test_scalars['EPE'], list) else \
            avg_test_scalars['EPE']

            # 更新最佳权重
            if current_epe < best_epe:
                best_epe = current_epe
                best_checkpoint_data = {
                    'epoch': epoch_idx,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_epe': best_epe,
                    'metrics': avg_test_scalars
                }
                torch.save(best_checkpoint_data, best_checkpoint_path)
                print(f"Updated best checkpoint (EPE: {best_epe:.4f}) saved to {best_checkpoint_path}")
            else:
                print(f"Current EPE ({current_epe:.4f}) not better than best ({best_epe:.4f}), no update")

            gc.collect()


def train_sample(sample, compute_metrics=False):
    model.train()
    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()
    optimizer.zero_grad()
    disp_ests = model(imgL, imgR)
    # SCARED数据集视差掩码调整（根据实际数据集视差范围修改）
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0) & (~torch.isnan(disp_gt)) & (~torch.isinf(disp_gt))

    # 根据训练模式选择损失函数
    if args.attention_weights_only:
        loss = model_loss_train_attn_only(disp_ests, disp_gt, mask)
    elif args.freeze_attention_weights:
        loss = model_loss_train_freeze_attn(disp_ests, disp_gt, mask)
    else:
        loss = model_loss_train(disp_ests, disp_gt, mask)

    scalar_outputs = {"loss": loss}
    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}

    if compute_metrics:
        with torch.no_grad():
            image_outputs["errormap"] = [disp_error_image_func.apply(disp_est, disp_gt) for disp_est in disp_ests]
            scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
            scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
            scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
            scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
            scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]

    loss.backward()
    optimizer.step()
    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


@make_nograd_func
def test_sample(sample, compute_metrics=True):
    model.eval()
    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()
    # SCARED测试集掩码调整
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0) & (~torch.isnan(disp_gt)) & (~torch.isinf(disp_gt))

    disp_ests = model(imgL, imgR)
    loss = model_loss_test(disp_ests, disp_gt, mask)

    scalar_outputs = {"loss": loss}
    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}
    image_outputs["errormap"] = [disp_error_image_func.apply(disp_est, disp_gt) for disp_est in disp_ests]
    scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
    scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
    scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


if __name__ == '__main__':
    train()