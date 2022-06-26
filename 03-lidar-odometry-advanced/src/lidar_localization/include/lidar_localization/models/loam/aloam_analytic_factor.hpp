#ifndef LIDAR_LOCALIZATION_MODELS_ALOAM_ANALYTIC_FACTOR_HPP_
#define LIDAR_LOCALIZATION_MODELS_ALOAM_ANALYTIC_FACTOR_HPP_
#include <eigen3/Eigen/Dense>
#include <sophus/so3.hpp>
#include <sophus/se3.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

class EdgeAnalyticFactor : public ceres::SizedCostFunction<3,4,3>{
//残差维度3，旋转四元数维度4，平移向量维度3
public:
//定义变量
	Eigen::Vector3d curr_point, last_point_a, last_point_b;
	double s;//去畸变参数默认为1
  //构造函数
  EdgeAnalyticFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_a_, Eigen::Vector3d last_point_b_, double s_)
		: curr_point(curr_point_), last_point_a(last_point_a_), last_point_b(last_point_b_), s(s_) {}

  //主体计算函数
  virtual bool Evaluate(double const* const* parameters,  
                        double *residuals, double** jacobians) const
{
  //定义两个优化变量
  Eigen::Map<const Eigen::Quaterniond> q_last_curr(parameters[0]);
  Eigen::Map<const Eigen::Vector3d> t_last_curr(parameters[1]);
  //当前点统一到上一帧的坐标系
  Eigen::Vector3d lp = q_last_curr*curr_point+t_last_curr;
  //残差公式计算
  Eigen::Vector3d nu = (lp-last_point_a).cross(lp-last_point_a);//分子
  Eigen::Vector3d de = last_point_a-last_point_b;//分母
  double nu_norm = nu.norm();
  double de_norm = de.norm();
  residuals[0] = nu_norm/de_norm;

  //雅可比计算
  if(jacobians!=nullptr && jacobians[0]!= nullptr){ ///为什么判断这两个值？
    //构建公式中的变量
    Eigen::Matrix3d skew_de = skew(de);
    Eigen::Vector3d rp = q_last_curr*curr_point;
    Eigen::Matrix3d skew_rp = skew(rp);
    //旋转雅可比计算
    Eigen::Map<Eigen::Matrix<double, 1, 4, Eigen::RowMajor>> J_so3(jacobians[0]);//RowMajor代表着行优先，在内存中，存储时按行存储
    J_so3.setZero();
    J_so3.block<1,3>(0,0) = nu.transpose() * skew_de*(-skew_rp)/(nu_norm*de_norm);
    //平移雅可比计算
    Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor>> J_t(jacobians[1]);
    J_t = nu.transpose() * skew_de/(nu_norm*de_norm);

  }

  return true;
}

};





#endif