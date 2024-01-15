#include <iostream>
#include <fstream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <boost/random.hpp>
#include <chrono>
#include <thread>
#include <filesystem>
#include "../include/saicp.hpp"

Eigen::Matrix4f generateRandomTransformation() {
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	boost::random::mt19937 gen(seed);  // 随机数生成器
	boost::random::uniform_real_distribution<> dis(-5, 5);  // 位移范围
	boost::random::uniform_real_distribution<> angle_dis(-M_PI, M_PI);  // 旋转范围

	Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
	transform(0, 3) = dis(gen); // X轴平移
	transform(1, 3) = dis(gen); // Y轴平移
	transform(2, 3) = dis(gen); // Z轴平移

	// Rotation disturbance around Z-axis
	float angleZ = angle_dis(gen);
	Eigen::Matrix4f rotZ = Eigen::Matrix4f::Identity();
	rotZ(0, 0) = cos(angleZ);
	rotZ(0, 1) = -sin(angleZ);
	rotZ(1, 0) = sin(angleZ);
	rotZ(1, 1) = cos(angleZ);

	// Rotation disturbance around Y-axis
	float angleY = angle_dis(gen);
	Eigen::Matrix4f rotY = Eigen::Matrix4f::Identity();
	rotY(0, 0) = cos(angleY);
	rotY(0, 2) = sin(angleY);
	rotY(2, 0) = -sin(angleY);
	rotY(2, 2) = cos(angleY);

	// Rotation disturbance around X-axis
	float angleX = angle_dis(gen);
	Eigen::Matrix4f rotX = Eigen::Matrix4f::Identity();
	rotX(1, 1) = cos(angleX);
	rotX(1, 2) = -sin(angleX);
	rotX(2, 1) = sin(angleX);
	rotX(2, 2) = cos(angleX);

	// Combine the transformations
	transform = transform * rotZ * rotY * rotX;
	return transform;
}

// 函数：随机选择一个.pcd文件
std::string selectRandomPCDFile(const std::string& directory) {
	std::vector<std::string> file_names;
	for (const auto& entry : std::filesystem::directory_iterator(directory)) {
		if (entry.path().extension() == ".pcd") {
			file_names.push_back(entry.path().filename().string());
		}
	}

	// 如果没有找到任何文件，则返回空字符串
	if (file_names.empty()) {
		return "";
	}

	// 随机选择一个文件
	srand(static_cast<unsigned int>(time(NULL)));  // 初始化随机数生成器
	std::string selected_file = file_names[rand() % file_names.size()];

	// 返回完整的文件路径
	return directory + selected_file;
}

void saveTransformation(const Eigen::Matrix4f &transform, const std::string &filename) {
	std::ofstream file(filename);
	if (file.is_open()) {
		file << transform;
		file.close();
	}
}

int main() {
	std::string directory = "/home/smile/Desktop/github/src/pointcloud-processing-visualization/pcd/";

	// 使用函数选择一个随机文件
	std::string file_to_load = selectRandomPCDFile(directory);

	// 加载点云
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>);
	if (pcl::io::loadPCDFile<pcl::PointXYZ>(file_to_load, *cloud_in) == -1) {
		PCL_ERROR("Couldn't read file\n");
		return -1;
	}

	// 移除NaN值
	std::vector<int> indices;
	pcl::removeNaNFromPointCloud(*cloud_in, *cloud_in, indices);

	// 进行体素滤波以减少点的数量
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
	voxel_grid.setInputCloud(cloud_in);
	voxel_grid.setLeafSize(0.07f, 0.07f, 0.07f);
	voxel_grid.filter(*cloud_filtered);

	// 随机生成一个初始变换
	Eigen::Matrix4f base_transformation = generateRandomTransformation();
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_transformed(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*cloud_filtered, *cloud_transformed, base_transformation);

	// 模拟退火参数
	double temperature = 5.2; // 初始温度
	double coolingRate = 0.99; // 冷却率
	int max_iterations = 1000; // 最大迭代次数

	// 创建SimulatedAnnealingICP实例
	SimulatedAnnealingICP saicp;
	saicp.setInputSource(cloud_transformed);
	saicp.setInputTarget(cloud_filtered);
	saicp.setInitialTemperature(temperature);
	saicp.setCoolingRate(coolingRate);
	saicp.setMaximumIterations(max_iterations);

	bool gasaicp_fitness_reached = false;
	pcl::PointCloud<pcl::PointXYZ>::Ptr alignedCloud(new pcl::PointCloud<pcl::PointXYZ>);
	saicp.align(alignedCloud);

	if (!gasaicp_fitness_reached) 
	{
		// 执行退火ICP
		std::cout << "========" << "GGGGGGGGG!" << "========\n";
		if (saicp.hasConverged()) 
		{
			double fitness_score = saicp.getFitnessScore();
			if (fitness_score <= 0.001) 
			{
				gasaicp_fitness_reached = true;
				std::cout << "退火ICP完成收敛\n";
				std::cout << "变换矩阵:\n" << saicp.getFinalTransformation() << "\n";
				std::cout << "初始矩阵:\n" << base_transformation << "\n" ;
			}
			else
			{
				std::cout << "退火ICP收敛失败, 请重试" <<	std::endl;
				return -1;
			}
		}
	}
	return 0;
}