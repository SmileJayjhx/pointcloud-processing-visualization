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

// 生成随机扰动(方便跳出循环)
Eigen::Matrix4f generateAnnealingTransformation(double temperature) {

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    boost::random::mt19937 gen(seed);
    double tras_scale = 0.15;
    double rot_scale = 0.17 * M_PI;
    boost::random::uniform_real_distribution<> rand_dis(0, 1);

    boost::random::uniform_real_distribution<> dis(-tras_scale * temperature, tras_scale * temperature);
    boost::random::uniform_real_distribution<> angle_dis(-rot_scale * temperature, rot_scale * temperature);

    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform(0, 3) = dis(gen); // X-axis translation disturbance
    transform(1, 3) = dis(gen); // Y-axis translation disturbance
    transform(2, 3) = dis(gen); // Z-axis translation disturbance

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


Eigen::Matrix4f generateRandomTransformation() {
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	boost::random::mt19937 gen(seed);  // 随机数生成器
	boost::random::uniform_real_distribution<> dis(-200, 200);  // 位移范围
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

struct TransformationError {
    float translationError;
    Eigen::Vector3f rotationError; // 存储绕X轴、Y轴和Z轴的旋转误差
};

// 重载 << 运算符以打印 TransformationError
std::ostream& operator<<(std::ostream& os, const TransformationError& error) {
    os << "Translation Error: " << error.translationError << ", "
       << "Rotation Error: [" << error.rotationError.transpose() << "]";
    return os;
}

// 示例：计算两个变换矩阵之间的误差
TransformationError CalculateTransformationError(const Eigen::Matrix4f &matrix1, const Eigen::Matrix4f &matrix2) {
    TransformationError error;

    // 计算平移误差
    Eigen::Vector3f translation1 = matrix1.block<3,1>(0, 3);
    Eigen::Vector3f translation2 = matrix2.block<3,1>(0, 3);
    error.translationError = (translation2 - translation1).norm();

    // 计算旋转误差
    Eigen::Quaternionf quaternion1(matrix1.block<3,3>(0,0));
    Eigen::Quaternionf quaternion2(matrix2.block<3,3>(0,0));
    Eigen::Quaternionf deltaQuaternion = quaternion1.inverse() * quaternion2;
    Eigen::Vector3f deltaEulerAngles = deltaQuaternion.toRotationMatrix().eulerAngles(0, 1, 2); // X, Y, Z
    error.rotationError = deltaEulerAngles.cwiseAbs(); // 绝对值误差

    return error;
}
Eigen::Matrix4f readMatrixFromFile(const std::string& filepath) {
    std::ifstream file(filepath);
    Eigen::Matrix4f matrix;

    if (file.is_open()) {
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                if (!(file >> matrix(i, j))) {
                    throw std::runtime_error("文件格式错误或数据不足以填充矩阵");
                }
            }
        }
        file.close();
    } else {
        throw std::runtime_error("无法打开文件: " + filepath);
    }

    return matrix;
}

int main() {
	// 配置退火时的随机数种子
	srand(static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count()));

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
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);

	// 进行体素滤波
	pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
	voxel_grid.setInputCloud(cloud_in);
	voxel_grid.setLeafSize(0.07f, 0.07f, 0.07f);  // 设置体素大小
	voxel_grid.filter(*cloud_filtered);

	// 模拟退火参数
	double temperature = 5.2; // 初始温度
	double coolingRate = 0.985; // 冷却率
	// 全局变量
	double last_fitness_score = std::numeric_limits<double>::max(); // 初始设置为最大值

	// 生成变换并保存到文件
	Eigen::Matrix4f base_transformation = generateRandomTransformation();
	Eigen::Matrix4f base_transformation_normal = base_transformation;
	std::cout << "Base Transformation Matrix:\n" << base_transformation << std::endl;
	saveTransformation(base_transformation, "/home/smile/Desktop/github/src/pointcloud-processing-visualization/saicp/result.txt");

	// 应用初始变换
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_transformed(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*cloud_filtered, *cloud_transformed, base_transformation);

	// 设置SAICP实例
	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> saicp, normal_icp;
	saicp.setInputSource(cloud_transformed);
	saicp.setInputTarget(cloud_filtered);
	saicp.setMaximumIterations(1); // 每次调用align时执行一次迭代

	normal_icp.setInputSource(cloud_transformed);
	normal_icp.setInputTarget(cloud_filtered);
	normal_icp.setMaximumIterations(1);

	// 初始化可视化
	pcl::visualization::PCLVisualizer viewer("SAICP demo");
	viewer.setBackgroundColor(0, 0, 0);
	viewer.addPointCloud<pcl::PointXYZ>(cloud_filtered, "cloud_filtered");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud_filtered");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "cloud_filtered");

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_saicp(new pcl::PointCloud<pcl::PointXYZ>(*cloud_transformed));
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_icp_normal(new pcl::PointCloud<pcl::PointXYZ>(*cloud_transformed));

	viewer.addPointCloud<pcl::PointXYZ>(cloud_saicp, "cloud_saicp");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud_saicp");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 1.0, 0.0, "cloud_saicp");

	viewer.addPointCloud<pcl::PointXYZ>(cloud_icp_normal, "cloud_icp_normal");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud_icp_normal");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 1.0, "cloud_icp_normal");

	viewer.addCoordinateSystem(1.0);
	viewer.initCameraParameters();

	// 创建初始变换的矩阵, 先验位姿的是一个单位阵, 即无先验位姿
	Eigen::Matrix4f saicp_result = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f normal_icp_result = Eigen::Matrix4f::Identity();

	// 计数器
	int saicp_cnt = 1; // icp迭代次数
	int normal_icp_cnt = 1; // 先验icp迭代次数

	bool saicp_fitness_reached = false; 
	bool normal_icp_fitness_reached = false;

	int iteration_counter = 0;  // 迭代频率计数器, 迭代的频率按照 10ms x iteration_counter 可以在下面的循环中修改

	int stop_iteration_cnt = 0;
	bool has_published = false;
	int bad_value_accetp_cnt = 0;
	while (!viewer.wasStopped()) {
		viewer.spinOnce();  
		// 图像化界面刷新频率10ms, 方便使用鼠标进行控制视角 
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
		// 如果都完成了收敛, 则不再更新
		if((saicp_fitness_reached && normal_icp_fitness_reached) || (++stop_iteration_cnt >= 2000)) 
		{
			if(!has_published)
			{
				double icp_score = saicp.getFitnessScore();
				double icp_normal_score = normal_icp.getFitnessScore();
				std::cout << "退火ICP迭代次数: " << saicp_cnt << " 退火ICP分数: " << icp_score <<std::endl;
				std::cout << "普通ICP迭代次数: " << normal_icp_cnt << " 普通ICP分数: " << icp_normal_score <<std::endl;
				std::cout << "迭代次数比率: " << (saicp_cnt-normal_icp_cnt)/normal_icp_cnt <<std::endl;
				std::cout << "分数差比率: " << std::abs((icp_score-icp_normal_score))/icp_normal_score <<std::endl;
				std::cout << "差值接收率: " << double(bad_value_accetp_cnt)/double(saicp_cnt) <<std::endl;
				std::cout << "[SAICP]变换矩阵 " << std::endl;
				std::cout << saicp.getFinalTransformation() << std::endl;
				std::cout << "[SAICP]误差 " << std::endl;
				Eigen::Matrix4f result = readMatrixFromFile("/home/smile/Desktop/github/src/pointcloud-processing-visualization/saicp/result.txt");
				std::cout << CalculateTransformationError(saicp.getFinalTransformation(),result) <<std::endl;
				std::cout << "-----------------------------------------------------------" << std::endl;
				std::cout << "[SAICP]变换矩阵 " <<std::endl;
				std::cout << normal_icp.getFinalTransformation() << std::endl;
				std::cout << "[SAICP]误差 " << std::endl;
				std::cout << CalculateTransformationError(normal_icp.getFinalTransformation(),result) <<std::endl;
				std::cout << "-----------------------------------------------------------" << std::endl;
				has_published = true;
			}
			continue;
		}
		
		// 创建icp之后的新点云
		pcl::PointCloud<pcl::PointXYZ>::Ptr saicp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr normal_icp_cloud(new pcl::PointCloud<pcl::PointXYZ>);

		// 每10ms x 100 = 1000ms = 1s 即每1秒做一次icp并更新点云
		if (++iteration_counter >= 2) {

			// 如果没有达到0.0001的分值, 则icp继续迭代
			if(!saicp_fitness_reached)
			{
				// 在原有变换的基础上添加模拟退火的随机扰动
				Eigen::Matrix4f annealing_transform = generateAnnealingTransformation(temperature);
				Eigen::Matrix4f perturbed_transformation = saicp_result * annealing_transform;

				// 应用带有扰动的变换进行ICP迭代
				saicp.align(*saicp_cloud, perturbed_transformation);
				Eigen::Matrix4f new_icp_result = saicp.getFinalTransformation();

				// 检查是否收敛(肯定收敛, 因为最多迭代1次,所以每一次都会收敛)
				if (saicp.hasConverged()) 
				{
					// 退火
					double new_fitness_score = saicp.getFitnessScore();

					if (new_fitness_score < last_fitness_score) 
					{
						saicp_result = new_icp_result; // 接受更好的变换
						last_fitness_score = new_fitness_score; // 更新最新的fitness score
					} 
					else if (exp((-(new_fitness_score - last_fitness_score)) / temperature) > ((double)rand() / RAND_MAX))
					{
						bad_value_accetp_cnt++;	
						saicp_result = perturbed_transformation; // 以一定概率接受较差的变换
						last_fitness_score = new_fitness_score; // 更新fitness score，即使它变差了
					}
					// 更新温度
					temperature *= coolingRate;
					// std::cout << "======================================================="<<std::endl;
					// std::cout << "当前温度: " << temperature <<std::endl;
					// std::cout << "======================================================="<<std::endl;

					double fitness_score = saicp.getFitnessScore();
					if(saicp_fitness_reached) saicp_cnt=saicp_cnt;
					else saicp_cnt += 1;
					// std::cout << "[ICP] 分数为 " << fitness_score <<std::endl;

					// 获取最新一次的变换, 并将该变换应用到带先验的点云上, 更新该点云
					base_transformation = saicp.getFinalTransformation().cast<float>();
					pcl::transformPointCloud(*cloud_transformed, *saicp_cloud, base_transformation);
					viewer.updatePointCloud<pcl::PointXYZ>(saicp_cloud, "cloud_saicp");
					//真正的停止条件(收敛条件)
					if(fitness_score<=0.001)
					{
						saicp_fitness_reached = true;
						std::cout << "======================================================="<<std::endl;
						std::cout << "[ICP]完成收敛 " <<std::endl;
						std::cout << "[ICP]迭代次数为 " << saicp_cnt <<std::endl;
						std::cout << "[ICP]变换矩阵 " << std::endl;
						std::cout << saicp.getFinalTransformation() << std::endl;
						std::cout << "======================================================="<<std::endl;
					}     
				}
			}   

			// 普通icp    
			if(!normal_icp_fitness_reached)
			{
				normal_icp.align(*normal_icp_cloud, normal_icp_result);
				normal_icp_result = normal_icp.getFinalTransformation();
				// 同理, 这里并不是真正的停止条件
				if (normal_icp.hasConverged()) 
				{
					double fitness_score_normal = normal_icp.getFitnessScore();
					if(normal_icp_fitness_reached) normal_icp_cnt = normal_icp_cnt;
					else normal_icp_cnt += 1;
					// std::cout << "[ICP+先验] 分数为 " << fitness_score_normal <<std::endl;

					// 带先验的停止条件也是0.0001分以下终止
					if(fitness_score_normal<=0.001)
					{
					normal_icp_fitness_reached = true;
					std::cout << "======================================================="<<std::endl;
					std::cout << "[ICP+先验]完成收敛 " <<std::endl;
					std::cout << "[ICP+先验]迭代次数为 " << normal_icp_cnt <<std::endl;
					std::cout << "[ICP+先验]变换矩阵 " <<std::endl;
					std::cout << normal_icp.getFinalTransformation() << std::endl;
					std::cout << "======================================================="<<std::endl;
					}
					// 获取最新一次的变换, 并将该变换应用到带先验的点云上, 更新该点云
					base_transformation_normal = normal_icp.getFinalTransformation().cast<float>();
					pcl::transformPointCloud(*cloud_transformed, *normal_icp_cloud, base_transformation_normal);
					viewer.updatePointCloud<pcl::PointXYZ>(normal_icp_cloud, "cloud_icp_normal"); 
				}
			}
			// 重置迭代计数器
			iteration_counter = 0;
		}
	}
return 0;
}