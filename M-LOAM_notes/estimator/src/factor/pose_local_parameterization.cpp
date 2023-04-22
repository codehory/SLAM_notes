/*******************************************************
 * Copyright (C) 2020, RAM-LAB, Hong Kong University of Science and Technology
 *
 * This file is part of M-LOAM (https://ram-lab.com/file/jjiao/m-loam).
 * If you use this code, please cite the respective publications as
 * listed on the above websites.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Jianhao JIAO (jiaojh1994@gmail.com)
 *******************************************************/

#include "pose_local_parameterization.h"

void PoseLocalParameterization::setParameter()
{
    is_degenerate_ = false;
    V_update_ = Eigen::Matrix<double, 6, 6>::Identity();
}

/*!
 * @brief  重载的Plus函数给出了四元素+三平移量的更新方法
 * @param x  优化前的参数
 * @param delta 增量
 * @param x_plus_delta 更新后的参数
 * @return
 */
// state update
// description of update rule: LIC-Fusion: LiDAR-Inertial-Camera Odometry, IROS 2019
// description of solution remapping: On Degeneracy of Optimization-based State Estimation Problems, ICRA 2016
// The pointer coeffs must reference the four coefficients of Quaternion in the following order: *coeffs == {x, y, z, w}
bool PoseLocalParameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const
{
    Eigen::Map<const Eigen::Vector3d> p(x);
    Eigen::Map<const Eigen::Quaterniond> q(x + 3);

    Eigen::Map<const Eigen::Matrix<double, 6, 1> > dx(delta); // dx = [dp, dq]
    Eigen::Matrix<double, 6, 1> dx_update = V_update_ * dx;
    Eigen::Vector3d dp(dx_update.head<3>());
    // deltaQ是实现角度到四元数组的变换
    Eigen::Quaterniond dq = Utility::deltaQ(dx_update.tail<3>());

    // Eigen::Map<const Eigen::Vector3d> dp(delta);
    // Eigen::Quaterniond dq = Utility::deltaQ(Eigen::Map<const Eigen::Vector3d>(delta + 3)); // using theta to approximate q

    Eigen::Map<Eigen::Vector3d> p_plus(x_plus_delta);
    Eigen::Map<Eigen::Quaterniond> q_plus(x_plus_delta + 3);
    p_plus = p + dp;
    q_plus = (q * dq).normalized(); // q = _q * [0.5*delta, 1]

    return true;
}

/*!  calculate the jacobian of [p, q] w.r.t [dp, dq]
 *   turtial: https://blog.csdn.net/hzwwpgmwy/article/details/86490556
 *   j =  1 0 0 0 0 0
 *        0 1 0 0 0 0
 *        0 0 1 0 0 0
 *        0 0 0 1 0 0
 *        0 0 0 0 1 0
 *        0 0 0 0 0 1
 *        0 0 0 0 0 0
 *
 *   7行6列 对应于7D姿态，前6个对应于3D旋转和平移，最后一个分量表示比例因子的标量。其中每一行对应一个维度应用于姿势的扰动，每一列对应于局部参数化空间的一个维度。
 *   雅可比行列式的前6行设置为单位矩阵，这意味着姿态的6个维度中的任何一个扰动（3个用于旋转，3个用于平移）将导致局部参数化相同维度的相应扰动空间。
 *   雅可比行列式的底行设置为零，这意味着姿势的比例因子的扰动不会导致局部参数化空间中的任何扰动。
 */

bool PoseLocalParameterization::ComputeJacobian(const double *x, double *jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor> > j(jacobian);
    j.topRows<6>().setIdentity();
    j.bottomRows<1>().setZero();
    return true;
}




//
