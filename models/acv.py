from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from models.submodule import *
import math
import gc
import time

# 特征提取模块（feature extraction）
# 功能：从输入图像中提取三级特征图l2 l3 l4（1/4分辨率），拼接后输出320通道特征，提取多尺度特征，用于后续构建代价体积
# 设计逻辑：类ResNet架构，通过初始卷积，生成不同层级的语义/纹理特征
class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()
        
        self.inplanes = 32 # 初始通道数（对应论文初始卷积输出32通道）
        # 初始下采样：3个3*3卷积+ReLU，将输入3通道（RGB）转换成32通道
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),#论文中提到特征提取的初始层采用步长为2的卷积，实现下采样并减少计算量
                                       nn.ReLU(inplace=True), # 激活函数，增强非线性能力
                                       convbn(32, 32, 3, 1, 1, 1), # 保持通道数不变
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1), # 进一步细化特征
                                       nn.ReLU(inplace=True))
        # 残差块组：采用BasicBlock构建，对应不同尺度的特征提取
        # 论文中设计多尺度特征提取以捕捉不同层级的图像信息（纹理、语义等）
        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1) # 输出通道32，3个残差块，步长1
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1) # 输出通道64，16个残差块，步长2（下采样）
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1) # 输出通道128，3个残差块
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2) # 扩张卷积（dilation=2），扩大感受野

    # 构建残差块组：根据输入参数生成由多个BasicBlock组成的序列
    # 对应论文中残差网络的设计，通过下采样和膨胀卷积平衡感受野与分辨率
    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None # 下采样模块（当输入输出通道或步长不匹配时使用）
        # 若步长不为1或输入通道与输出通道（planes*expansion）不匹配，需要下采样调整
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False), # 1x1卷积调整通道数和分辨率
                nn.BatchNorm2d(planes * block.expansion), # 批归一化稳定训练
            )

        layers = []
        # 添加第一个残差块（可能包含下采样）
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion # 更新输入通道数
        # 添加剩余残差块（步长为1，无下采样）
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers) # 返回残差块序列

    # 前向传播：提取多尺度特征并拼接，用于构建GWC体积，对应论文中多尺度特征融合策略，拼接不同层级特征以保留丰富信息
    def forward(self, x):
        x = self.firstconv(x)# 初始卷积：输出32通道
        x = self.layer1(x) # 第一层残差块：输出32通道
        l2 = self.layer2(x) # 第二层残差块：输出64通道（下采样后）
        l3 = self.layer3(l2) # 第三层残差块：输出128通道
        l4 = self.layer4(l3) # 第四层残差块：输出128通道（膨胀卷积）
        # 拼接l2（64）、l3（128）、l4（128），总通道320，作为GWC体积的输入特征
        gwc_feature = torch.cat((l2, l3, l4), dim=1)
        return {"gwc_feature": gwc_feature} # 返回字典形式的特征

# 沙漏网络：用于代价体积的细化，通过下采样-上采样路径保留多尺度上下文信息,对应论文3.4节Cost Volume Refinement部分：采用沙漏结构增强代价体积的判别性
class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        # 下采样路径：两次卷积+批归一化+ReLU，通道数翻倍，分辨率减半
        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1),  # 3D卷积：输入通道→2*输入通道，步长2（下采样）
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1), # 3D卷积：保持通道和分辨率
                                   nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1), # 再次下采样，通道数→4*输入通道
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1), # 保持通道和分辨率
                                   nn.ReLU(inplace=True))

        # 自注意力块：增强特征相关性，对应论文中3.3节的注意力机制，捕捉长距离依赖
        self.attention_block = attention_block(channels_3d=in_channels * 4, num_heads=16, block=(4, 4, 4))

        # 上采样路径：转置卷积实现上采样，恢复分辨率
        self.conv5 = nn.Sequential(
            # 3D转置卷积：4*输入通道→2*输入通道，步长2（上采样），输出填充1以匹配分辨率
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2)) # 批归一化

        self.conv6 = nn.Sequential(
            # 3D转置卷积：2*输入通道→输入通道，步长2（上采样至原始分辨率）
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        # 跳跃连接：保留下采样前的特征，缓解信息丢失（对应沙漏网络的经典设计）
        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    # 前向传播：通过下采样、注意力增强、上采样和跳跃连接细化代价体积
    def forward(self, x):
        conv1 = self.conv1(x)  # 下采样1：通道×2，分辨率÷2
        conv2 = self.conv2(conv1)  # 特征细化
        conv3 = self.conv3(conv2)  # 下采样2：通道×2，分辨率÷2
        conv4 = self.conv4(conv3)  # 特征细化
        conv4 = self.attention_block(conv4)  # 注意力增强：突出有效特征，抑制噪声
        # 上采样1 + 跳跃连接：与conv2残差融合
        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        # 上采样2 + 跳跃连接：与输入x残差融合，恢复原始分辨率和通道
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)
        return conv6  # 返回细化后的代价体积

# ACVNet主网络：整合特征提取、注意力权重生成、ACV构建和视差回归。对应论文图1整体架构：实现Attention Concatenation Volume的端到端立体匹配
class ACVNet(nn.Module):
    def __init__(self, maxdisp, attn_weights_only, freeze_attn_weights):
        super(ACVNet, self).__init__()
        self.maxdisp = maxdisp # 最大视差范围（论文中设置为192）
        self.attn_weights_only = attn_weights_only # 是否仅训练注意力权重（分阶段训练第一步）
        self.freeze_attn_weights = freeze_attn_weights # 是否冻结注意力权重（分阶段训练第二步）
        self.num_groups = 40 # GWC体积的分组数（论文3.3节：分组相关计算减少冗余）
        self.concat_channels = 32 # 拼接特征的通道数（用于构建拼接体积）
        self.feature_extraction = feature_extraction()# 特征提取网络实例

        # 拼接特征处理卷积：将320通道（l2+l3+l4）降至32通道，用于构建拼接体积
        self.concatconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1), # 3x3卷积降维至128通道
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, self.concat_channels, kernel_size=1, padding=0, stride=1,# 1x1卷积至32通道
                                                    bias=False))

        # 多尺度上下文建模卷积（对应论文3.3节：对GWC体积应用不同膨胀率的卷积，捕捉多尺度上下文）
        self.patch = nn.Conv3d(40, 40, kernel_size=(1,3,3), stride=1, dilation=1, groups=40, padding=(0,1,1), bias=False)
        # 分组处理：将40通道GWC体积分为3组，分别应用不同膨胀率（1,2,3）的卷积
        self.patch_l1 = nn.Conv3d(8, 8, kernel_size=(1,3,3), stride=1, dilation=1, groups=8, padding=(0,1,1), bias=False)
        self.patch_l2 = nn.Conv3d(16, 16, kernel_size=(1,3,3), stride=1, dilation=2, groups=16, padding=(0,2,2), bias=False)
        self.patch_l3 = nn.Conv3d(16, 16, kernel_size=(1,3,3), stride=1, dilation=3, groups=16, padding=(0,3,3), bias=False)

        # 注意力权重生成网络（对应论文3.3节Attention Weight Generation）
        self.dres1_att_ = nn.Sequential(convbn_3d(40, 32, 3, 1, 1),# 初始3D卷积块
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        # 沙漏网络细化注意力特征
        self.dres2_att_ = hourglass(32)

        # 输出注意力权重（单通道）
        self.classif_att_ = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        # ACV体积处理网络（用于视差回归，对应论文3.4节Cost Volume Refinement）
        # 初始卷积块：处理ACV体积（32×2=64通道）
        self.dres0 = nn.Sequential(convbn_3d(self.concat_channels * 2, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        # 残差块+沙漏网络：细化代价体积
        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))        
        self.dres2 = hourglass(32) # 第一个沙漏网络

        self.dres3 = hourglass(32) # 第二个沙漏网络（进一步细化）

        # 视差预测头（多尺度输出，用于监督训练，对应论文中的多阶段损失）
        self.classif0 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))
        # 从第一个沙漏输出预测视差
        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))
        # 从第二个沙漏输出预测视差
        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        # 权重初始化（论文中未特别提及，但标准做法是正态分布初始化卷积权重，批归一化权重设1）
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n)) # 正态分布初始化，方差为2/n
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)# 批归一化权重初始化为1
                m.bias.data.zero_()# 偏置初始化为0
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    # 前向传播：输入左右图像，输出视差预测。对应论文图1的流程：特征提取→GWC体积→注意力权重→ACV体积→视差回归
    def forward(self, left, right):
        # 第一步：生成注意力权重（分阶段训练：是否冻结权重）
        if self.freeze_attn_weights:
            # 冻结注意力权重时，用torch.no_grad()停止梯度计算
            with torch.no_grad():
                # 提取左右图像特征
                features_left = self.feature_extraction(left)
                features_right = self.feature_extraction(right)
                # 构建GWC体积（Group-Wise Correlation Volume），对应论文3.3节
                gwc_volume = build_gwc_volume(features_left["gwc_feature"], features_right["gwc_feature"], self.maxdisp // 4, self.num_groups)
                # 多尺度上下文建模：对GWC体积应用不同膨胀率的卷积
                gwc_volume = self.patch(gwc_volume)
                patch_l1 = self.patch_l1(gwc_volume[:, :8]) # 前8通道：dilation=1
                patch_l2 = self.patch_l2(gwc_volume[:, 8:24]) # 中间16通道：dilation=2
                patch_l3 = self.patch_l3(gwc_volume[:, 24:40]) # 后16通道：dilation=3
                patch_volume = torch.cat((patch_l1,patch_l2,patch_l3), dim=1)# 拼接多尺度特征
                # 注意力权重生成：通过卷积和沙漏网络
                cost_attention = self.dres1_att_(patch_volume)
                cost_attention = self.dres2_att_(cost_attention)
                att_weights = self.classif_att_(cost_attention) # 输出注意力权重

        else:
            # 不冻结时，正常计算梯度（训练注意力权重或联合训练）
            features_left = self.feature_extraction(left)
            features_right = self.feature_extraction(right)
            gwc_volume = build_gwc_volume(features_left["gwc_feature"], features_right["gwc_feature"], self.maxdisp // 4, self.num_groups)
            gwc_volume = self.patch(gwc_volume)
            patch_l1 = self.patch_l1(gwc_volume[:, :8])
            patch_l2 = self.patch_l2(gwc_volume[:, 8:24])
            patch_l3 = self.patch_l3(gwc_volume[:, 24:40])
            patch_volume = torch.cat((patch_l1,patch_l2,patch_l3), dim=1)
            cost_attention = self.dres1_att_(patch_volume)
            cost_attention = self.dres2_att_(cost_attention)
            att_weights = self.classif_att_(cost_attention)

        # 第二步：构建ACV体积并回归视差（若不仅训练注意力权重）
        if not self.attn_weights_only:
            # 处理拼接特征（左右图像）
            concat_feature_left = self.concatconv(features_left["gwc_feature"])
            concat_feature_right = self.concatconv(features_right["gwc_feature"])
            # 构建拼接体积（Concatenation Volume），对应论文3.3节
            concat_volume = build_concat_volume(concat_feature_left, concat_feature_right, self.maxdisp // 4)
            # 注意力加权：ACV = softmax(注意力权重) × 拼接体积（论文核心创新点）
            ac_volume = F.softmax(att_weights, dim=2) * concat_volume   ### ac_volume = att_weights * concat_volume
            # 代价体积细化与视差预测
            cost0 = self.dres0(ac_volume)  # 初始处理
            cost0 = self.dres1(cost0) + cost0  # 残差连接
            out1 = self.dres2(cost0)  # 第一个沙漏网络输出
            out2 = self.dres3(out1)  # 第二个沙漏网络输出

        # 训练阶段：返回多尺度视差预测（用于多阶段损失计算）
        if self.training:

            if not self.freeze_attn_weights:
                # 上采样注意力权重至原始分辨率，用于计算注意力损失
                # cost_attention = F.upsample(att_weights, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
                cost_attention = F.interpolate(att_weights, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
                cost_attention = torch.squeeze(cost_attention, 1)
                pred_attention = F.softmax(cost_attention, dim=1)
                pred_attention = disparity_regression(pred_attention, self.maxdisp)

            if not self.attn_weights_only:

                cost0 = self.classif0(cost0)
                cost1 = self.classif1(out1)
                cost2 = self.classif2(out2)    

                cost0 = F.upsample(cost0, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
                cost0 = torch.squeeze(cost0, 1)
                pred0 = F.softmax(cost0, dim=1)
                pred0 = disparity_regression(pred0, self.maxdisp)    

                cost1 = F.upsample(cost1, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
                cost1 = torch.squeeze(cost1, 1)
                pred1 = F.softmax(cost1, dim=1)
                pred1 = disparity_regression(pred1, self.maxdisp)    

                cost2 = F.upsample(cost2, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
                cost2 = torch.squeeze(cost2, 1)
                pred2 = F.softmax(cost2, dim=1)
                pred2 = disparity_regression(pred2, self.maxdisp)

                if self.freeze_attn_weights:
                    return [pred0, pred1, pred2]
                return [pred_attention, pred0, pred1, pred2]
            return [pred_attention]

        else:

            if self.attn_weights_only:

                cost_attention = F.upsample(att_weights, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
                cost_attention = torch.squeeze(cost_attention, 1)
                pred_attention = F.softmax(cost_attention, dim=1)
                pred_attention = disparity_regression(pred_attention, self.maxdisp)
                return [pred_attention]

            cost2 = self.classif2(out2)
            # cost2 = F.upsample(cost2, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost2 = F.interpolate(cost2, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = disparity_regression(pred2, self.maxdisp)

            return [pred2]

def acv(d):
    return ACVNet(d)
