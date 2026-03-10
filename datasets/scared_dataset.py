import os
import random
from torch.utils.data import Dataset  # 导入PyTorch的Dataset基类
from PIL import Image  # 用于图像加载和处理
import numpy as np  # 用于数值运算和数组处理
from datasets.data_io import get_transform, read_all_lines, pfm_imread  # 导入自定义数据工具函数
import tifffile  # 导入库


class ScaredDatset(Dataset):
    """
    SCARED 数据集加载类，用于读取立体图像对和对应的视差图（disparity）
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
        # 适配SCARED图像尺寸（宽1280×高1024）
        self.img_width = 1280
        self.img_height = 1024

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
        # return Image.open(filename).convert('RGB')
        """加载图像并转换为RGB格式（保持与原接口一致）"""
        # return Image.open(os.path.join(self.datapath, filename)).convert('RGB')
        img_path = os.path.normpath(os.path.join(self.datapath, filename))  # 统一路径格式
        return Image.open(img_path).convert('RGB')



    def load_disp(self, filename):
        """加载.tiff格式视差图（适配SCARED数据集格式）"""
        disp_path = os.path.join(self.datapath, filename)
        # 用tifffile读取Tiff文件（支持16位/32位深度）
        disp = tifffile.imread(disp_path)
        # 转为float32并确保内存连续
        disp = np.array(disp, dtype=np.float32)
        disp = np.ascontiguousarray(disp)
        return disp
        # # 读取tiff格式视差图，转为float32类型
        # disp = Image.open(os.path.join(self.datapath, filename))
        # disp = np.array(disp, dtype=np.float32)
        # # 确保数组内存连续，避免后续运算错误
        # disp = np.ascontiguousarray(disp)
        # return disp

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
        # print(f"Loading sample index: {index}")  # 新增

        # 加载左图、右图、视差图（拼接完整路径）
        left_img = self.load_image(self.left_filenames[index])
        right_img = self.load_image(self.right_filenames[index])
        disparity = self.load_disp(self.disp_filenames[index])

        if self.training:
            # 训练模式：随机裁剪+标准化（适配1280×1024输入）
            crop_w, crop_h = 640, 480  # 合理裁剪尺寸，兼顾模型输入和视差范围
            # 随机生成裁剪左上角坐标（确保不超出原图范围）
            x1 = random.randint(0, self.img_width - crop_w)
            y1 = random.randint(0, self.img_height - crop_h)
            # 裁剪图像
            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            # 裁剪视差图（numpy数组切片）
            disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]
            # 图像标准化（复用ACVNet的get_transform函数，保持一致性）
            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)
            return {
                "left": left_img,
                "right": right_img,
                "disparity": disparity
            }

        else:
            # 测试模式：固定裁剪+标准化+保留元信息（便于结果对齐）
            # crop_w, crop_h = 1280, 960  # 测试时尽量保留完整图像，仅裁剪底部冗余
            crop_w, crop_h = 1280, 1024  # 测试时尽量保留完整图像，仅裁剪底部冗余
            # 从右下角裁剪（固定位置，确保测试一致性）
            x1 = self.img_width - crop_w
            y1 = self.img_height - crop_h
            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]
            # 图像标准化
            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)
            # 返回测试所需完整信息（与原接口一致，兼容后续评估流程）
            return {
                "left": left_img,
                "right": right_img,
                "disparity": disparity,
                "top_pad": 0,  # 无顶部填充，保持接口兼容
                "right_pad": 0,  # 无右侧填充，保持接口兼容
                "left_filename": self.left_filenames[index]  # 原始文件名，用于结果保存
            }