# 导入PyTorch的函数式接口（包含常用损失函数、激活函数等）
import torch.nn.functional as F
# 导入PyTorch库
import torch


def model_loss_train_attn_only(disp_ests, disp_gt, mask):
    """
    仅训练注意力模块时使用的损失函数（训练阶段）

    参数:
        disp_ests (list[torch.Tensor]): 模型输出的视差估计值列表（可能包含多个尺度的输出，此处仅用1个）
        disp_gt (torch.Tensor): 视差真实值（ground truth）
        mask (torch.Tensor): 掩码张量，用于过滤无效像素（如遮挡区域、边界外区域等），仅计算有效区域的损失

    返回:
        torch.Tensor: 加权求和后的总损失
    """
    # 损失权重列表，此处仅1个权重（对应单输出的视差估计）
    weights = [1.0]
    # 存储各输出层的损失
    all_losses = []
    # 遍历视差估计值和对应权重（此处仅1组）
    for disp_est, weight in zip(disp_ests, weights):
        # 计算有效区域（mask为True的区域）的Smooth L1损失，再乘以权重后加入列表
        # Smooth L1损失对异常值更稳健（相比L1损失，在误差较小时接近L2，误差较大时接近L1）
        # size_average=True表示对损失求平均值
        all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True))
    # 返回所有损失的总和
    return sum(all_losses)


def model_loss_train_freeze_attn(disp_ests, disp_gt, mask):
    """
    冻结注意力模块时使用的损失函数（训练阶段）

    参数:
        disp_ests (list[torch.Tensor]): 模型输出的视差估计值列表（包含3个尺度的输出）
        disp_gt (torch.Tensor): 视差真实值（ground truth）
        mask (torch.Tensor): 掩码张量，用于过滤无效像素

    返回:
        torch.Tensor: 加权求和后的总损失
    """
    # 损失权重列表，3个权重分别对应3个尺度的视差估计（权重随输出层级递增，强调高层输出的损失）
    weights = [0.5, 0.7, 1.0]
    all_losses = []
    # 遍历3个尺度的视差估计和对应权重
    for disp_est, weight in zip(disp_ests, weights):
        # 计算有效区域的Smooth L1损失并加权
        all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True))
    return sum(all_losses)


def model_loss_train(disp_ests, disp_gt, mask):
    """
    常规训练（不特殊处理注意力模块）时使用的损失函数（训练阶段）

    参数:
        disp_ests (list[torch.Tensor]): 模型输出的视差估计值列表（包含4个尺度的输出）
        disp_gt (torch.Tensor): 视差真实值（ground truth）
        mask (torch.Tensor): 掩码张量，用于过滤无效像素

    返回:
        torch.Tensor: 加权求和后的总损失
    """
    # 损失权重列表，4个权重分别对应4个尺度的视差估计（权重递增，高层输出权重更高）
    weights = [0.5, 0.5, 0.7, 1.0]
    all_losses = []
    # 遍历4个尺度的视差估计和对应权重
    for disp_est, weight in zip(disp_ests, weights):
        # 计算有效区域的Smooth L1损失并加权
        all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True))
    return sum(all_losses)


def model_loss_test(disp_ests, disp_gt, mask):
    """
    测试阶段使用的损失函数（评估模型性能）

    参数:
        disp_ests (list[torch.Tensor]): 模型输出的视差估计值列表（仅使用1个最终输出）
        disp_gt (torch.Tensor): 视差真实值（ground truth）
        mask (torch.Tensor): 掩码张量，用于过滤无效像素

    返回:
        torch.Tensor: 计算得到的L1损失（评估指标）
    """
    # 损失权重列表，仅1个权重（对应最终输出的视差估计）
    weights = [1.0]
    all_losses = []
    # 遍历视差估计值和对应权重（此处仅1组）
    for disp_est, weight in zip(disp_ests, weights):
        # 测试阶段使用L1损失（直接计算绝对误差的平均值），更直观反映估计精度
        all_losses.append(weight * F.l1_loss(disp_est[mask], disp_gt[mask], size_average=True))
    return sum(all_losses)