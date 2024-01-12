#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <Eigen/Core>
#include <boost/random.hpp>
#include <fstream>
#include <chrono>
#include <string>

class SimulatedAnnealingICP {
public:
    SimulatedAnnealingICP();
    void setInputSource(pcl::PointCloud<pcl::PointXYZ>::Ptr source);
    void setInputTarget(pcl::PointCloud<pcl::PointXYZ>::Ptr target);
    void setInitialTemperature(double temp);
    void setCoolingRate(double rate);
    void setMaximumIterations(int max_iter);
    void align(pcl::PointCloud<pcl::PointXYZ>::Ptr &output_cloud);
    bool hasConverged() const;
    double getFitnessScore() const;
    Eigen::Matrix4f getFinalTransformation() const;

private:
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target_;
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp_;
    double initial_temperature_;
    double cooling_rate_;
    int max_iterations_;
    Eigen::Matrix4f final_transformation_;
    double fitness_score_;
    bool converged_;

    Eigen::Matrix4f generateAnnealingTransformation(double temperature);
};


