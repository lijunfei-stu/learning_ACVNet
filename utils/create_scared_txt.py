import os

# 生成SCARED数据集的训练文件列表
# 参数:
# root_dir: SCARED数据集根目录（即C:\Users\12700\Desktop\All_datasets\SCARED）
# output_file: 输出的
# txt文件路径（如. / filenames / SCARED_train.txt）
def generate_train_txt(root_dir, output_file):
    # 存储所有数据行
    lines = []

    # 遍历TRAIN下的所有dataset（如dataset_1, dataset_2...）
    # train_dir = os.path.join(root_dir, 'TRAIN')
    train_dir = os.path.join(root_dir, 'TEST1')
    for dataset in os.listdir(train_dir):
        dataset_path = os.path.join(train_dir, dataset)
        if not os.path.isdir(dataset_path):
            continue  # 跳过非文件夹

        # 遍历每个dataset下的keyframe（如keyframe_1, keyframe_2...）
        for keyframe in os.listdir(dataset_path):
            keyframe_path = os.path.join(dataset_path, keyframe)
            if not os.path.isdir(keyframe_path):
                continue  # 跳过非文件夹

            # 定义数据路径
            data_path = os.path.join(keyframe_path, 'data')
            left_dir = os.path.join(data_path, 'left_finalpass')
            right_dir = os.path.join(data_path, 'right_finalpass')
            disp_dir = os.path.join(data_path, 'disparity')

            # 检查路径是否存在
            if not all(os.path.exists(p) for p in [left_dir, right_dir, disp_dir]):
                continue  # 跳过缺少数据的keyframe

            # 获取所有左图文件名（假设左右图和视差图文件名一一对应）
            left_files = [f for f in os.listdir(left_dir) if f.endswith('.png')]
            for left_file in left_files:
                # 提取文件名（不含扩展名）
                base_name = os.path.splitext(left_file)[0]

                # 构建对应右图和视差图的文件名
                right_file = f"{base_name}.png"
                disp_file = f"{base_name}.tiff"

                # 检查右图和视差图是否存在
                if not os.path.exists(os.path.join(right_dir, right_file)):
                    print(f"警告：未找到右图 {right_file}，已跳过")
                    continue
                if not os.path.exists(os.path.join(disp_dir, disp_file)):
                    print(f"警告：未找到视差图 {disp_file}，已跳过")
                    continue

                # 构建相对路径（相对于root_dir）
                left_rel_path = os.path.relpath(os.path.join(left_dir, left_file), root_dir)
                right_rel_path = os.path.relpath(os.path.join(right_dir, right_file), root_dir)
                disp_rel_path = os.path.relpath(os.path.join(disp_dir, disp_file), root_dir)

                # 替换Windows路径分隔符为Unix风格（避免路径错误）
                left_rel_path = left_rel_path.replace('\\', '/')
                right_rel_path = right_rel_path.replace('\\', '/')
                disp_rel_path = disp_rel_path.replace('\\', '/')

                # 添加到列表
                lines.append(f"{left_rel_path} {right_rel_path} {disp_rel_path}")

    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"成功生成 {output_file}，共包含 {len(lines)} 条数据")


if __name__ == '__main__':
    # 配置路径
    ROOT_DIR = "C:\\Users\\12700\\Desktop\\All_datasets\\SCARED"  # 数据集根目录
    # OUTPUT_FILE = r"../filenames/SCARED_train.txt" # 输出文件路径（需提前创建filenames文件夹）
    OUTPUT_FILE = r"../filenames/SCARED1_test.txt" # 输出文件路径（需提前创建filenames文件夹）

    # 生成文件列表
    generate_train_txt(ROOT_DIR, OUTPUT_FILE)