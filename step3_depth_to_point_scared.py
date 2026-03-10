import json
import numpy as np
from PIL import Image
import os
import open3d as o3d


def read_camera_parameters(json_path):
    """读取相机参数（基线、焦距、主点坐标）"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    calib = data['camera-calibration']
    KL = np.array(calib['KL'])  # 左相机内参矩阵

    # 提取内参参数 (fx, fy, cx, cy)
    fx = KL[0, 0]  # x方向焦距（像素）
    fy = KL[1, 1]  # y方向焦距（像素）
    cx = KL[0, 2]  # 主点x坐标（像素）
    cy = KL[1, 2]  # 主点y坐标（像素）

    # 提取基线（单位转换为米）
    T = np.array(calib['T']).flatten()
    baseline = abs(T[0]) / 1000.0  # 基线长度（米）

    return baseline, fx, fy, cx, cy


def depth_to_pointcloud(depth_path, fx, fy, cx, cy):
    """仅通过深度图生成点云（无颜色）"""
    # 读取深度图
    depth_img = Image.open(depth_path)
    depth = np.array(depth_img, dtype=np.float32)  # 深度值单位：米

    # 获取图像尺寸
    h, w = depth.shape[:2]

    # 生成像素坐标网格
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u = u.flatten()
    v = v.flatten()

    # 过滤无效深度值
    mask = depth.flatten() > 1e-6  # 忽略深度为0或接近0的点
    z = depth.flatten()[mask]
    u_valid = u[mask]
    v_valid = v[mask]

    # 计算三维坐标（相机坐标系）
    x = (u_valid - cx) * z / fx
    y = (v_valid - cy) * z / fy
    z = z  # 深度值即z坐标

    # 构建点云数据 (仅包含三维坐标)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.column_stack([x, y, z]))

    return pcd


def disp_to_depth(disp_path, baseline, fx, output_path):
    """将视差图转换为深度图（复用原有逻辑）"""
    disp_img = Image.open(disp_path)
    disp = np.array(disp_img, dtype=np.float32)
    mask = disp > 1e-6
    depth = np.zeros_like(disp)
    depth[mask] = (baseline * fx) / disp[mask]
    depth_img = Image.fromarray(depth.astype(np.float32))
    depth_img.save(output_path)
    print(f"深度图已保存至: {output_path}")
    return depth


if __name__ == '__main__':
    # 输入文件路径（请根据实际情况修改）
    disp_path = r"C:\Users\12700\Desktop\Self-sewing\others\ACVNet\data\scared\disp_tiff\data_frame_data000000_depth.tiff"
    # disp_path = r"C:\Users\12700\Desktop\All_datasets\SCARED\TEST\dataset_2\keyframe_1\data\disparity\frame_data000000.tiff"
    json_path = r"C:\Users\12700\Desktop\All_datasets\SCARED\TEST\dataset_2\keyframe_1\data\frame_data\frame_data000000.json"

    # 输出深度图路径
    output_dir = os.path.dirname(disp_path)
    output_name = os.path.splitext(os.path.basename(disp_path))[0].replace('disp', 'depth') + '.tiff'
    output_path = os.path.join(output_dir, output_name)

    # 1. 生成深度图
    baseline, fx, fy, cx, cy = read_camera_parameters(json_path)
    print(f"相机参数 - 基线: {baseline:.6f}m, fx: {fx:.2f}, fy: {fy:.2f}, cx: {cx:.2f}, cy: {cy:.2f}")
    disp_to_depth(disp_path, baseline, fx, output_path)

    # 2. 生成并显示点云（无颜色）
    pcd = depth_to_pointcloud(output_path, fx, fy, cx, cy)

    # 添加坐标系辅助显示
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    # 可视化点云（默认颜色）
    o3d.visualization.draw_geometries([pcd, coordinate_frame], window_name="深度图转点云（无颜色）")