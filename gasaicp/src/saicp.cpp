# include "../include/saicp.hpp"
#include <pcl/visualization/pcl_visualizer.h>
#include <thread>

SimulatedAnnealingICP::SimulatedAnnealingICP()
	: initial_temperature_(5.2), cooling_rate_(0.985), max_iterations_(500),
	fitness_score_(std::numeric_limits<double>::max()), converged_(false) 
{
	// 初始化随机数种子, 用于生成退火接收的比率
	srand(static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count()));
	// 默认构造函数实现，初始化变量
}

// 设置输入源点云
void SimulatedAnnealingICP::setInputSource(pcl::PointCloud<pcl::PointXYZ>::Ptr source) 
{
	cloud_source_ = source;
}

// 设置目标点云
void SimulatedAnnealingICP::setInputTarget(pcl::PointCloud<pcl::PointXYZ>::Ptr target) 
{
	cloud_target_ = target;
}

// 设置初始温度
void SimulatedAnnealingICP::setInitialTemperature(double temp) 
{
	initial_temperature_ = temp;
}

// 设置冷却率
void SimulatedAnnealingICP::setCoolingRate(double rate) 
{
	cooling_rate_ = rate;
}

// 设置最大迭代次数
void SimulatedAnnealingICP::setMaximumIterations(int max_iter) 
{
	max_iterations_ = max_iter;
}

// 执行对齐
void SimulatedAnnealingICP::align(pcl::PointCloud<pcl::PointXYZ>::Ptr &output_cloud) 
{
	// 初始化结果矩阵为单位矩阵
	Eigen::Matrix4f gasaicp_result = Eigen::Matrix4f::Identity();
	double last_fitness_score = std::numeric_limits<double>::max();
	double temperature = initial_temperature_;

	int iteration_counter = 0;
	converged_ = false;
	icp_.setInputSource(cloud_source_);
	icp_.setInputTarget(cloud_target_);
	pcl::PointCloud<pcl::PointXYZ>::Ptr tem_cloud(new pcl::PointCloud<pcl::PointXYZ>);

	// 初始化可视化
	pcl::visualization::PCLVisualizer viewer("SAICP demo");
	viewer.setBackgroundColor(0, 0, 0);
	viewer.addPointCloud<pcl::PointXYZ>(cloud_target_, "cloud_target_");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud_target_");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "cloud_target_");
	viewer.addCoordinateSystem(1.0);
	viewer.initCameraParameters();

	viewer.addPointCloud<pcl::PointXYZ>(cloud_source_, "saicp_icp");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "saicp_icp");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 1.0, 0.0, "saicp_icp");
	int bad_cnt = 1;
	while (!converged_) {
		// 在原有变换的基础上添加模拟退火的随机扰动
		Eigen::Matrix4f annealing_transform = generateAnnealingTransformation(temperature);
		Eigen::Matrix4f perturbed_transformation = gasaicp_result * annealing_transform;
		// 应用带有扰动的变换进行ICP迭代
		icp_.setMaximumIterations(1);
		icp_.align(*tem_cloud, perturbed_transformation);
		Eigen::Matrix4f new_icp_result = icp_.getFinalTransformation();

		if(icp_.hasConverged()) 
		{
			viewer.spinOnce();
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
			double new_fitness_score = icp_.getFitnessScore();
			double rand_num = static_cast<double>(rand()) / RAND_MAX;

			if(new_fitness_score < last_fitness_score) 
			{
				gasaicp_result = new_icp_result;
				last_fitness_score = new_fitness_score;
			} else if(exp((-(new_fitness_score - last_fitness_score)) / temperature) > rand_num) 
			{
				last_fitness_score = new_fitness_score;
				// std::cout << "接收差值, " << exp((-(new_fitness_score - last_fitness_score)) / temperature) << " 随机数: "<< rand_num << endl; 
				gasaicp_result = perturbed_transformation;
				bad_cnt++;
			}

			Eigen::Matrix4f saicp_final = icp_.getFinalTransformation().cast<float>();
			pcl::transformPointCloud(*cloud_source_, *tem_cloud, saicp_final);
			viewer.updatePointCloud<pcl::PointXYZ>(tem_cloud, "saicp_icp");

			temperature *= cooling_rate_;
			// std::cout << "current temp: " << temperature << std::endl;
		}
		// std::cout << "迭代次数: " << iteration_counter <<std::endl;
		iteration_counter++;
		if (iteration_counter > max_iterations_ || last_fitness_score<0.0001) {
			output_cloud = tem_cloud;
			converged_ = true;
			std::cout << "saipc has converged! " << std::endl;
			break;
		}
	}

	final_transformation_ = gasaicp_result;
	fitness_score_ = last_fitness_score;
}


// 检查是否已收敛
bool SimulatedAnnealingICP::hasConverged() const 
{
	return converged_;
}

// 获取适应度分数
double SimulatedAnnealingICP::getFitnessScore() const 
{
	return fitness_score_;
}

// 获取最终变换矩阵
Eigen::Matrix4f SimulatedAnnealingICP::getFinalTransformation() const 
{
	return final_transformation_;
}

// 生成退火变换
Eigen::Matrix4f SimulatedAnnealingICP::generateAnnealingTransformation(double temperature) 
{
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	boost::random::mt19937 gen(seed);
	double tras_scale = 0.15;
	double rot_scale = 0.17 * M_PI;
	boost::random::uniform_real_distribution<> rand_dis(0, 1);
	boost::random::uniform_real_distribution<> dis(-tras_scale * temperature, tras_scale * temperature);
	boost::random::uniform_real_distribution<> angle_dis(-rot_scale * temperature, rot_scale * temperature);

	Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
	// std::cout << "平移扰动大小: " << dis(gen) <<std::endl;
	transform(0, 3) = dis(gen); // X轴平移扰动
	transform(1, 3) = dis(gen); // Y轴平移扰动
	transform(2, 3) = dis(gen); // Z轴平移扰动

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

