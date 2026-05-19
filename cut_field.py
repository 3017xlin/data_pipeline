import numpy as np
import glob
import os
import concurrent.futures
import time

def process_single_file(filepath):
    """单个 Worker 负责处理一个 .npz 文件的过滤"""
    filename = os.path.basename(filepath)
    
    try:
        # 1. 读取数据
        data = np.load(filepath)
        data_dict = {key: data[key] for key in data.files}
        
        centers = data_dict['volume_mesh_centers']
        fields = data_dict['volume_fields']
        
        # 2. 提取 X, Y, Z 坐标
        x = centers[:, 0]
        y = centers[:, 1]
        z = centers[:, 2]
        
        # 3. 创建过滤掩码：保留 X[-550, 550], Y[-550, 550], Z[0, 160]
        mask = (x >= -550) & (x <= 550) & \
               (y >= -550) & (y <= 550) & \
               (z >= 0) & (z <= 160)
        
        # 4. 应用掩码
        filtered_centers = centers[mask]
        filtered_fields = fields[mask]
        
        # 5. 更新字典
        data_dict['volume_mesh_centers'] = filtered_centers
        data_dict['volume_fields'] = filtered_fields
        
        # 6. 另存为新文件 (_filtered.npz)
        new_filepath = filepath.replace('.npz', '_filtered.npz')
        np.savez_compressed(new_filepath, **data_dict)
        
        orig_len = len(centers)
        new_len = len(filtered_centers)
        
        return f"[成功] {filename} -> 剔除了 {orig_len - new_len} 个点."
        
    except Exception as e:
        return f"[失败] {filename} 发生错误: {e}"


def run_parallel_filtering(folder_path, max_workers=10):
    """主控函数：分配任务给各个核心"""
    search_pattern = os.path.join(folder_path, '*.npz')
    all_files = glob.glob(search_pattern)
    # 只处理原始文件，避开已经带有 '_filtered' 后缀的文件
    files_to_process = [f for f in all_files if '_filtered' not in f]
    
    if not files_to_process:
        print("没有找到需要处理的原始 .npz 文件！(可能都已经过滤过了)")
        return

    total_files = len(files_to_process)
    print(f"🚀 共找到 {total_files} 个文件准备过滤瘦身...")
    print(f"🔥 启动多进程引擎，当前设置的 Worker 数量: {max_workers}\n")

    start_time = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_file, filepath): filepath for filepath in files_to_process}
        
        completed_count = 0
        for future in concurrent.futures.as_completed(futures):
            completed_count += 1
            result_msg = future.result()
            print(f"({completed_count}/{total_files}) {result_msg}")

    end_time = time.time()
    print(f"\n✅ 所有文件过滤完毕！总耗时: {end_time - start_time:.2f} 秒.")


if __name__ == '__main__':
    # 你的文件存放路径
    target_folder = r"C:\Users\3017y\Downloads\streamline" 
    
    # 运行过滤，使用 10 个核心
    run_parallel_filtering(target_folder, max_workers=10)