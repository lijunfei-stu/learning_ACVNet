import cv2
import json
import numpy as np
from os.path import join

def read_Q(reprojection_file):
    """从JSON文件中读取重投影矩阵Q"""
    with open(reprojection_file) as json_file:
        data = json.load(json_file)
        Q = data['reprojection-matrix']
        return np.array(Q)

def disp_to_depth(disp_img, Q):
    """
    将视差图转换为深度图
    参数:
        disp_img: 视差图数据
        Q: 重投影矩阵
    返回:
        depth: 计算得到的深度图
    """
    # 从重投影矩阵解析参数
    fl = Q[2, 3]  # 焦距
    bl = 1 / Q[3, 2]  # 基线距离
    print(f'焦距: {fl}, 基线: {bl}')

    # 初始化深度图
    depth = np.zeros_like(disp_img, dtype=np.float32)

    # 遍历视差图计算深度 (深度 = 焦距 * 基线 / 视差)
    # 注意过滤视差为0的无效像素
    valid_mask = disp_img > 0
    depth[valid_mask] = (fl * bl) / disp_img[valid_mask]

    return depth


def process_single_disp(disp_path, q_path, output_depth_path):
    """处理单个视差图并保存深度图"""
    # 读取视差图 (使用IMREAD_ANYDEPTH确保读取原始深度值)
    disp_img = cv2.imread(disp_path, cv2.IMREAD_ANYDEPTH)
    if disp_img is None:
        raise FileNotFoundError(f"无法读取视差图: {disp_path}")

    # 读取重投影矩阵
    Q = read_Q(q_path)

    # 转换为深度图
    depth_map = disp_to_depth(disp_img, Q)

    # 保存深度图 (使用TIFF格式保留精度)
    cv2.imwrite(output_depth_path, depth_map)
    print(f"深度图已保存至: {output_depth_path}")

    # 可选：显示结果（如需可视化可取消注释）
    # cv2.imshow("Disparity", disp_img / disp_img.max())  # 归一化显示
    # cv2.imshow("Depth", depth_map / depth_map[depth_map > 0].max())  # 排除0值归一化
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def visualize_depth(depth_path, save_path=None, cmap=cv2.COLORMAP_JET):
    """
    读取深度图并可视化

    参数:
        depth_path: 深度图文件路径
        save_path: 可选，可视化结果保存路径，为None时不保存
        cmap: 颜色映射表，默认使用JET颜色映射
    """
    # 读取深度图（支持float32等格式）
    depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
    if depth_img is None:
        raise FileNotFoundError(f"无法读取深度图: {depth_path}")

    # 处理无效值（通常为0）
    valid_mask = depth_img > 0
    if not np.any(valid_mask):
        raise ValueError("深度图中没有有效深度值")

    # 提取有效深度值并归一化到0-255范围
    valid_depth = depth_img[valid_mask]
    min_depth = valid_depth.min()
    max_depth = valid_depth.max()

    # 归一化（排除无效值影响）
    normalized_depth = np.zeros_like(depth_img, dtype=np.uint8)
    normalized_depth[valid_mask] = 255 * (valid_depth - min_depth) / (max_depth - min_depth)

    # 应用颜色映射
    colored_depth = cv2.applyColorMap(normalized_depth, cmap)

    # 将无效区域标记为红色
    colored_depth[~valid_mask] = [0, 0, 255]  # BGR格式，红色

    # 显示深度图信息
    print(f"深度图尺寸: {depth_img.shape}")
    print(f"有效深度范围: {min_depth:.2f} - {max_depth:.2f}")
    print(f"无效像素数量: {np.sum(~valid_mask)}")

    # 调整窗口大小便于查看
    cv2.namedWindow("Depth Visualization", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Depth Visualization", 800, 600)

    # 显示图像
    cv2.imshow("Depth Visualization", colored_depth)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存可视化结果
    if save_path:
        cv2.imwrite(save_path, colored_depth)
        print(f"可视化结果已保存至: {save_path}")


if __name__ == '__main__':
    # 单个文件处理参数（请根据实际路径修改）
    # 视差图路径
    disp_file = r'C:\Users\12700\Desktop\All_datasets\SCARED\TRAIN\dataset_2\keyframe_1\data\disparity\frame_data000000.tiff' # todo 真实
    # disp_file = r'C:\Users\12700\Desktop\Self-sewing\others\ACVNet\data\scared\disp_raw\data_frame_data000087.tiff' # todo 预测
    # 对应的重投影矩阵JSON文件路径
    q_file = r'C:\Users\12700\Desktop\All_datasets\SCARED\TRAIN\dataset_2\keyframe_1\data\reprojection_data\frame_data000000.json'
    # 输出深度图路径
    output_depth = r'./data/scared/depth/test_depth_data000087.tiff'

    # 执行转换
    # process_single_disp(disp_file, q_file, output_depth)

    depth_map = cv2.imread(output_depth, cv2.IMREAD_ANYDEPTH)
    cv2.imshow("Depth", depth_map / depth_map[depth_map > 0].max())  # 排除0值归一化
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 可视化深度图
    # visualize_depth(
    #     depth_path=output_depth,
    #     save_path='',  # 可选保存路径
    #     cmap=cv2.COLORMAP_MAGMA  # 可更换颜色映射（如COLORMAP_PLASMA, COLORMAP_MAGMA等）
    # )
