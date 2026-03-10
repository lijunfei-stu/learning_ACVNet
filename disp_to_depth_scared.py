import json
import numpy as np
from PIL import Image
import os


def read_camera_parameters(json_path):
    """读取相机参数并计算基线（转换为米）和焦距"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    calib = data['camera-calibration']

    # 提取左相机内参的x方向焦距（像素）
    KL = np.array(calib['KL'])
    fx = KL[0, 0]  # x方向焦距（像素）

    # 提取平移向量（基线），根据双目腔镜参数可知单位为毫米，需转换为米
    T = np.array(calib['T']).flatten()
    baseline_mm = abs(T[0])  # 基线长度（毫米），取x方向平移的绝对值
    baseline = baseline_mm / 1000.0  # 转换为米

    return baseline, fx


def disp_to_depth(disp_path, baseline, fx, output_path):
    """将视差图转换为深度图"""
    # 读取视差图（float32格式）
    disp_img = Image.open(disp_path)
    disp = np.array(disp_img, dtype=np.float32)

    # 避免除零错误（视差为0的区域深度设为0）
    mask = disp > 1e-6  # 忽略接近0的视差
    depth = np.zeros_like(disp)

    # 深度计算公式：Z = (基线 * 焦距) / 视差
    depth[mask] = (baseline * fx) / disp[mask]

    # 保存深度图为TIFF格式（float32）
    depth_img = Image.fromarray(depth.astype(np.float32))
    depth_img.save(output_path)
    print(f"深度图已保存至: {output_path}")
    print(f"深度图统计信息 - 最小值: {np.min(depth[mask]):.4f}m, 最大值: {np.max(depth[mask]):.4f}m")


if __name__ == '__main__':
    # 输入文件路径（可根据实际情况修改）
    disp_path = r"C:\Users\12700\Desktop\Self-sewing\others\ACVNet\data\scared\disp_tiff\data_frame_data000000.tiff"
    json_path = r"C:\Users\12700\Desktop\All_datasets\SCARED\TEST\dataset_2\keyframe_1\data\frame_data\frame_data000000.json"

    # 输出深度图路径
    output_dir = os.path.dirname(disp_path)
    output_name = os.path.splitext(os.path.basename(disp_path))[0].replace('disp', 'depth') + '.tiff'
    output_path = os.path.join(output_dir, output_name)

    # 执行转换
    baseline, fx = read_camera_parameters(json_path)
    print(f"相机参数 - 基线长度: {baseline:.6f}m ({baseline * 1000:.6f}mm), 焦距: {fx:.2f}像素")
    disp_to_depth(disp_path, baseline, fx, output_path)