import os
import glob

# 设置源文件夹和目标文件夹
src_directory = '/home/smile/Desktop/github/src/pointcloud-processing-visualization/pcd/'
dest_directory = src_directory  # 目标目录和源目录相同

# 获取所有子目录中的.pcd文件
pcd_files = glob.glob(os.path.join(src_directory, '**/*.pcd'), recursive=True)

# 移动并重命名文件
for i, file in enumerate(pcd_files):
    # 生成新的文件名
    new_name = os.path.join(dest_directory, f"{i+1}.pcd")

    # 移动文件到目标目录并重命名
    os.rename(file, new_name)
    print(f"Moved and renamed '{file}' to '{new_name}'")
