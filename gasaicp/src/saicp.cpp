# include "../include/saicp.hpp"

SimulatedAnnealingICP::SimulatedAnnealingICP()
	: initial_temperature_(5.2), cooling_rate_(0.985), max_iterations_(500),
	fitness_score_(std::numeric_limits<double>::max()), converged_(false) {
	// 初始化随机数种子, 用于生成退火接收的比率
	srand(static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count()));
	// 默认构造函数实现，初始化变量
}

// 设置输入源点云
void SimulatedAnnealingICP::setInputSource(pcl::PointCloud<pcl::PointXYZ>::Ptr source) {
	cloud_source_ = source;
}

// 设置目标点云
void SimulatedAnnealingICP::setInputTarget(pcl::PointCloud<pcl::PointXYZ>::Ptr target) {
	cloud_target_ = target;
}

// 设置初始温度
void SimulatedAnnealingICP::setInitialTemperature(double temp) {
	initial_temperature_ = temp;
}

// 设置冷却率
void SimulatedAnnealingICP::setCoolingRate(double rate) {
	cooling_rate_ = rate;
}

// 设置最大迭代次数
void SimulatedAnnealingICP::setMaximumIterations(int max_iter) {
	max_iterations_ = max_iter;
}

// 执行对齐
void SimulatedAnnealingICP::align(pcl::PointCloud<pcl::PointXYZ>::Ptr &output_cloud) {
	// 初始化结果矩阵为单位矩阵
	Eigen::Matrix4f gasaicp_result = Eigen::Matrix4f::Identity();
	double last_fitness_score = std::numeric_limits<double>::max();
	double temperature = initial_temperature_;

	int iteration_counter = 0;
	converged_ = false;
	while (temperature > 0.0001 && iteration_counter < max_iterations_) {
		// 在原有变换的基础上添加模拟退火的随机扰动
		Eigen::Matrix4f annealing_transform = generateAnnealingTransformation(temperature);
		Eigen::Matrix4f perturbed_transformation = gasaicp_result * annealing_transform;

		// 应用带有扰动的变换进行ICP迭代
		icp_.setInputSource(cloud_source_);
		icp_.setInputTarget(cloud_target_);
		icp_.setMaximumIterations(1);
		icp_.align(*cloud_source_, gasaicp_result);

		if (icp_.hasConverged()) {
			double new_fitness_score = icp_.getFitnessScore();
			if (new_fitness_score < last_fitness_score) {
				gasaicp_result = icp_.getFinalTransformation();
				last_fitness_score = new_fitness_score;
			} else if (exp((-(last_fitness_score - new_fitness_score)) / temperature) > static_cast<double>(rand()) / RAND_MAX) {
				gasaicp_result = perturbed_transformation;
				last_fitness_score = new_fitness_score;
			}

			temperature *= cooling_rate_;
			std::cout << "current temp: " << temperature << std::endl;
		}

		iteration_counter++;
		if (iteration_counter > max_iterations_ || last_fitness_score<0.0001) {
			*output_cloud = *cloud_source_;
			converged_ = true;
			break;
		}
	}

	final_transformation_ = gasaicp_result;
	fitness_score_ = last_fitness_score;
}


// 检查是否已收敛
bool SimulatedAnnealingICP::hasConverged() const {
	return converged_;
}

// 获取适应度分数
double SimulatedAnnealingICP::getFitnessScore() const {
	return fitness_score_;
}

// 获取最终变换矩阵
Eigen::Matrix4f SimulatedAnnealingICP::getFinalTransformation() const {
	return final_transformation_;
}

// 生成退火变换
Eigen::Matrix4f SimulatedAnnealingICP::generateAnnealingTransformation(double temperature) {
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	boost::random::mt19937 gen(seed);
	double tras_scale = 0.15;
	double rot_scale = 0.17 * M_PI;
	boost::random::uniform_real_distribution<> rand_dis(0, 1);
	double randNum = static_cast<double>(rand()) / RAND_MAX;
	if (rand_dis(gen) > randNum && temperature <= 0.1) {
		tras_scale *= 1.2;
		rot_scale *= 1.7;
	}

	boost::random::uniform_real_distribution<> dis(-tras_scale * temperature, tras_scale * temperature);
	boost::random::uniform_real_distribution<> angle_dis(-rot_scale * temperature, rot_scale * temperature);

	Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
	transform(0, 3) = dis(gen); // X轴平移扰动
	transform(1, 3) = dis(gen); // Y轴平移扰动
	transform(2, 3) = dis(gen); // Z轴平移扰动

	// 绕Z轴旋转扰动
	float angle = angle_dis(gen);
	transform(0, 0) = cos(angle);
	transform(0, 1) = -sin(angle);
	transform(1, 0) = sin(angle);
	transform(1, 1) = cos(angle);

	return transform;
}

