#include <iostream>
#include <fstream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/gicp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <boost/random.hpp>
#include <chrono>
#include <thread>

Eigen::Matrix4f generateRandomTransformation() {
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	boost::random::mt19937 gen(seed);  // 随机数生成器
	boost::random::uniform_real_distribution<> dis(-0.5, 0.5);  // 位移范围
	boost::random::uniform_real_distribution<> angle_dis(-M_PI, M_PI);  // 旋转范围

	Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
	transform(0, 3) = dis(gen); // X轴平移
	transform(1, 3) = dis(gen); // Y轴平移
	transform(2, 3) = dis(gen); // Z轴平移

	// 绕Z轴旋转
	float angle = angle_dis(gen);
	transform(0, 0) = cos(angle);
	transform(0, 1) = -sin(angle);
	transform(1, 0) = sin(angle);
	transform(1, 1) = cos(angle);
	return transform;
}

void saveTransformation(const Eigen::Matrix4f &transform, const std::string &filename) {
	std::ofstream file(filename);
	if (file.is_open()) {
		file << transform;
		file.close();
	}
}

Eigen::Matrix4f generateCloseTransformation(const Eigen::Matrix4f &original) {
	// 单位 [m]
	Eigen::Matrix4f closeTransform = original;
	closeTransform(0, 3) += 0.05; // X轴微调
	closeTransform(1, 3) += 0.05; // Y轴微调
	closeTransform(2, 3) += 0.05; // Z轴微调

	// 单位 [rad]
	Eigen::Matrix3f rotation = closeTransform.block<3, 3>(0, 0);
	Eigen::AngleAxisf rotationX(0.05, Eigen::Vector3f::UnitX());
	Eigen::AngleAxisf rotationY(0.05, Eigen::Vector3f::UnitY());
	Eigen::AngleAxisf rotationZ(0.05, Eigen::Vector3f::UnitZ());
	rotation *= (rotationX * rotationY * rotationZ).matrix();
	closeTransform.block<3, 3>(0, 0) = rotation;
	return closeTransform;
}


int main() {
	// 加载点云
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>);
	if (pcl::io::loadPCDFile<pcl::PointXYZ>("/home/smile/Desktop/github/src/pointcloud-processing-visualization/pcd/gicp.pcd", *cloud_in) == -1) {
		PCL_ERROR("Couldn't read file test.pcd \n");
		return -1;
	}

	// 移除NaN值
	std::vector<int> indices;
	pcl::removeNaNFromPointCloud(*cloud_in, *cloud_in, indices);

	// 进行体素滤波
	pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
	voxel_grid.setInputCloud(cloud_in);
	voxel_grid.setLeafSize(0.08f, 0.08f, 0.08f);  // 设置体素大小
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
	voxel_grid.filter(*cloud_filtered);

	// 生成变换并保存到文件
	Eigen::Matrix4f base_transformation = generateRandomTransformation();
	Eigen::Matrix4f base_transformation_prior = base_transformation;
	std::cout << "Base Transformation Matrix:\n" << base_transformation << std::endl;
	saveTransformation(base_transformation, "/home/smile/Desktop/github/src/pointcloud-processing-visualization/gicp/result.txt");

	// 生成接近的变换作为先验位姿
	Eigen::Matrix4f prior_pose = generateCloseTransformation(base_transformation);

	// 应用初始变换
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_transformed(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*cloud_filtered, *cloud_transformed, base_transformation);

	// 设置gicp实例
	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp, gicp_with_prior;
	gicp.setInputSource(cloud_transformed);
	gicp.setInputTarget(cloud_filtered);
	gicp.setMaximumIterations(1); // 每次调用align时执行一次迭代

	gicp_with_prior.setInputSource(cloud_transformed);
	gicp_with_prior.setInputTarget(cloud_filtered);
	gicp_with_prior.setMaximumIterations(1);

	// 初始化可视化
	pcl::visualization::PCLVisualizer viewer("gicp demo");
	viewer.setBackgroundColor(0, 0, 0);
	viewer.addPointCloud<pcl::PointXYZ>(cloud_filtered, "cloud_filtered");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud_filtered");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "cloud_filtered");

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_gicp(new pcl::PointCloud<pcl::PointXYZ>(*cloud_transformed));
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_gicp_prior(new pcl::PointCloud<pcl::PointXYZ>(*cloud_transformed));

	viewer.addPointCloud<pcl::PointXYZ>(cloud_gicp, "cloud_gicp");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud_gicp");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 1.0, 0.0, "cloud_gicp");

	viewer.addPointCloud<pcl::PointXYZ>(cloud_gicp_prior, "cloud_gicp_prior");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud_gicp_prior");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 1.0, "cloud_gicp_prior");

	viewer.addCoordinateSystem(1.0);
	viewer.initCameraParameters();

	// 创建初始变换的矩阵, 其中无先验位姿的是一个单位阵
	Eigen::Matrix4f gicp_result = Eigen::Matrix4f::Identity();
	// 先验的矩阵为之前生成的, 在实际变换的基础上添加了扰动的变换矩阵
	Eigen::Matrix4f gicp_result_prior = prior_pose;

	// 计数器
	int gicp_cnt = 0; // gicp迭代次数
	int gicp_prior_cnt = 0; // 先验gicp迭代次数

	bool gicp_fitness_reached = false; 
	bool gicp_prior_fitness_reached = false;

	int iteration_counter = 0;  // 迭代频率计数器, 迭代的频率按照 10ms x iteration_counter 可以在下面的循环中修改

	while (!viewer.wasStopped()) {

		
		viewer.spinOnce();  
		// 图像化界面刷新频率10ms, 方便使用鼠标进行控制视角 
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
		// 如果都完成了收敛, 则不再更新
		if(gicp_fitness_reached && gicp_prior_fitness_reached) continue;
		// 创建gicp之后的新点云
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_gicp_it(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_gicp_prior_it(new pcl::PointCloud<pcl::PointXYZ>);

		// 每10ms x 100 = 1000ms = 1s 即每1秒做一次gicp并更新点云
		if (++iteration_counter >= 100) {

			// 如果没有达到0.0001的分值, 则gicp继续迭代
			if(!gicp_fitness_reached)
			{
				// 对普通gicp和带有先验位姿的gicp进行迭代
				gicp.align(*cloud_gicp_it,gicp_result);
				gicp_result = gicp.getFinalTransformation();

				// 检查是否收敛(肯定收敛, 因为最多迭代1次,所以每一次都会收敛)
				if (gicp.hasConverged()) 
				{
					double fitness_score = gicp.getFitnessScore();
					if(gicp_fitness_reached) gicp_cnt=gicp_cnt;
					else gicp_cnt += 1;
					std::cout << "[gicp] 分数为 " << fitness_score <<std::endl;

					// 获取最新一次的变换, 并将该变换应用到带先验的点云上, 更新该点云
					base_transformation = gicp.getFinalTransformation().cast<float>();
					pcl::transformPointCloud(*cloud_transformed, *cloud_gicp_it, base_transformation);
					viewer.updatePointCloud<pcl::PointXYZ>(cloud_gicp_it, "cloud_gicp");
					//真正的停止条件(收敛条件)
					if(fitness_score<=0.001)
					{
						gicp_fitness_reached = true;
						std::cout << "======================================================="<<std::endl;
						std::cout << "[gicp]完成收敛 " <<std::endl;
						std::cout << "[gicp]迭代次数为 " << gicp_cnt <<std::endl;
						std::cout << "[gicp]变换矩阵 " << std::endl;
						std::cout << gicp.getFinalTransformation() << std::endl;
						std::cout << "======================================================="<<std::endl;
					}     
				}
			}       
			if(!gicp_prior_fitness_reached)
			{
				gicp_with_prior.align(*cloud_gicp_prior_it, gicp_result_prior);
				gicp_result_prior = gicp_with_prior.getFinalTransformation();
				// 同理, 这里并不是真正的停止条件
				if (gicp_with_prior.hasConverged()) 
				{
					double fitness_score_prior = gicp_with_prior.getFitnessScore();
					if(gicp_prior_fitness_reached) gicp_prior_cnt = gicp_prior_cnt;
					else gicp_prior_cnt += 1;
					std::cout << "[gicp+先验] 分数为 " << fitness_score_prior <<std::endl;

					// 带先验的停止条件也是0.0001分以下终止
					if(fitness_score_prior<=0.001)
					{
					gicp_prior_fitness_reached = true;
					std::cout << "======================================================="<<std::endl;
					std::cout << "[gicp+先验]完成收敛 " <<std::endl;
					std::cout << "[gicp+先验]迭代次数为 " << gicp_prior_cnt <<std::endl;
					std::cout << "[gicp+先验]变换矩阵 " <<std::endl;
					std::cout << gicp_with_prior.getFinalTransformation() << std::endl;
					std::cout << "======================================================="<<std::endl;
					}
					// 获取最新一次的变换, 并将该变换应用到带先验的点云上, 更新该点云
					base_transformation_prior = gicp_with_prior.getFinalTransformation().cast<float>();
					pcl::transformPointCloud(*cloud_transformed, *cloud_gicp_prior_it, base_transformation_prior);
					viewer.updatePointCloud<pcl::PointXYZ>(cloud_gicp_prior_it, "cloud_gicp_prior"); 
				}
			}
			// 重置迭代计数器
			iteration_counter = 0;
		}
	}
return 0;
}