import os
import random
from torch.utils.data import Dataset  # 导入PyTorch的Dataset基类
from PIL import Image  # 用于图像加载和处理
import numpy as np  # 用于数值运算和数组处理
from datasets.data_io import get_transform, read_all_lines, pfm_imread  # 导入自定义数据工具函数

# 960*540
class SceneFlowDatset(Dataset):
    """
    Scene Flow 数据集加载类，用于读取立体图像对和对应的视差图（disparity）
    继承自PyTorch的Dataset，需实现__len__和__getitem__方法
    """

    def __init__(self, datapath, list_filename, training):
        """
        初始化数据集

        参数:
            datapath: 数据集根目录路径
            list_filename: 包含图像路径的列表文件（每行存储左图、右图、视差图路径）
            training: 布尔值，True表示训练模式，False表示测试模式
        """
        self.datapath = datapath  # 存储数据集根目录
        # 从列表文件中加载左图、右图、视差图的路径列表
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(list_filename)
        self.training = training  # 标记训练/测试模式

    def load_path(self, list_filename):
        """
        从列表文件中读取图像路径

        参数:
            list_filename: 存储路径的文本文件路径

        返回:
            left_images: 左图路径列表
            right_images: 右图路径列表
            disp_images: 视差图路径列表
        """
        # 读取文件中所有行（去除换行符）
        lines = read_all_lines(list_filename)
        # 按空格分割每行，得到左图、右图、视差图路径
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]  # 提取左图路径
        right_images = [x[1] for x in splits]  # 提取右图路径
        disp_images = [x[2] for x in splits]  # 提取视差图路径
        return left_images, right_images, disp_images

    def load_image(self, filename):
        """
        加载图像并转换为RGB格式

        参数:
            filename: 图像文件路径

        返回:
            PIL.Image对象（RGB格式）
        """
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        """
        加载视差图（Scene Flow的视差图为.pfm格式）

        参数:
            filename: 视差图文件路径（.pfm格式）

        返回:
            视差图数据（numpy数组，float32类型）
        """
        # 调用pfm_imread读取.pfm文件，返回数据和缩放因子
        data, scale = pfm_imread(filename)
        # 转换为连续内存的float32数组（避免后续运算中的内存问题）
        data = np.ascontiguousarray(data, dtype=np.float32)
        return data

    def __len__(self):
        """
        返回数据集样本数量

        返回:
            数据集中样本的总数
        """
        return len(self.left_filenames)

    def __getitem__(self, index):
        """
        按索引获取一个样本（左图、右图、视差图），并根据训练/测试模式进行预处理

        参数:
            index: 样本索引

        返回:
            字典，包含预处理后的左图、右图、视差图（训练模式），或额外包含图像路径等信息（测试模式）
        """
        # 加载左图、右图、视差图（拼接完整路径）
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))
        disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]))

        if self.training:
            # 训练模式：随机裁剪+标准化

            # 获取图像原始尺寸
            w, h = left_img.size
            # 裁剪目标尺寸（宽512，高256）
            crop_w, crop_h = 512, 256

            # 随机生成裁剪的左上角坐标（确保裁剪区域在图像内）
            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)

            # 对左图、右图进行裁剪
            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            # 对视差图进行对应裁剪（numpy数组切片）
            disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]

            # 图像转换：转为Tensor并标准化（使用ImageNet的均值和标准差）
            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            # 返回训练样本（左图、右图、视差图）
            return {
                "left": left_img,
                "right": right_img,
                "disparity": disparity
            }
        else:
            # 测试模式：固定裁剪+标准化+保留额外信息

            # 获取图像原始尺寸
            w, h = left_img.size
            # 裁剪目标尺寸（宽960，高512）
            crop_w, crop_h = 960, 512

            # 从图像右下角裁剪（固定位置，确保一致性）
            left_img = left_img.crop((w - crop_w, h - crop_h, w, h))
            right_img = right_img.crop((w - crop_w, h - crop_h, w, h))
            # 视差图对应裁剪
            disparity = disparity[h - crop_h:h, w - crop_w: w]

            # 图像转换：转为Tensor并标准化
            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            # 返回测试样本（额外包含填充信息和文件名，用于后续结果对齐和保存）
            return {
                "left": left_img,
                "right": right_img,
                "disparity": disparity,
                "top_pad": 0,  # 顶部填充量（此处未填充，为0）
                "right_pad": 0,  # 右侧填充量（此处未填充，为0）
                "left_filename": self.left_filenames[index]  # 左图原始路径（用于结果命名）
            }