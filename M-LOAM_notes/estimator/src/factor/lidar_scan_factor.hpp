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

#pragma once

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <eigen3/Eigen/Dense>

#include "../utility/utility.h"

// ****************************************************************
// calculate distrance from point to plane (using normal)
class LidarScanPlaneNormFactor : public ceres::SizedCostFunction<1, 7>
{
public:
    LidarScanPlaneNormFactor(const Eigen::Vector3d &point,
                             const Eigen::Vector4d &coeff,
                             const double &s = 1.0)
        : point_(point), coeff_(coeff), s_(s) {}

    bool Evaluate(double const *const *param, double *residuals, double **jacobians) const
    {
        Eigen::Quaterniond q_last_curr(param[0][6], param[0][3], param[0][4], param[0][5]);
        Eigen::Vector3d t_last_curr(param[0][0], param[0][1], param[0][2]);
        q_last_curr = Eigen::Quaterniond::Identity().slerp(s_, q_last_curr);
        t_last_curr = s_ * t_last_curr;

        //平面法向量
        Eigen::Vector3d w(coeff_(0), coeff_(1), coeff_(2));
        //直线方程中的D
        double d = coeff_(3);
        //a = Ax+By+Cz+D
        double a = w.dot(q_last_curr * point_ + t_last_curr) + d;
        //残差为点到平面的距离
        residuals[0] = a;

        if (jacobians)
        {
            //将四元数转化为旋转矩阵
            Eigen::Matrix3d R = q_last_curr.toRotationMatrix();
            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> jacobian_pose(jacobians[0]);
                Eigen::Matrix<double, 1, 6> jaco; // [da/dt, da/dR, 1] 分别对平移量和旋转矩阵求偏导
                jaco.setZero();
                /*!
                 * a = w.dot(q_last_curr * point_ + t_last_curr) + d 对平移量求偏导
                 * da/dx = w.dot(d(q_last_curr * point_ + t_last_curr)/dx)  q_last_curr和d常量与x无关
                 *       = w.dot(d(t_last_curr)/dx)  t_last_curr--> (x, y, z)
                 *       = w.dot(1, 0, 0)
                 *       = wx
                 *同理：
                 * da/dy = w.dot(0, 1, 0) = wy
                 * da/dz = w.dot(0, 0, 1) = wz
                 *
                 * da/dt = w.transpose()
                 *
                 *
                 *
                 */

                jaco.leftCols<3>() = w.transpose();
                //Utility::skewSymmetric(point_)表示将向量point_转化为反对称矩阵,其作用是用于叉乘运算
                // 即 skewSymmetric(point_)= 0 -v3  v2
                //                          v3  0  -v1
                //                         -v2  v1  0

                //R的导数 = R* 旋转矢量的反对称矩阵

                /*! https://zhuanlan.zhihu.com/p/156895046
                 *   a = w.dot(q_last_curr * point_ + t_last_curr) + d 对旋转矩阵求偏导
                 *   da/dR = w.dot(d(q_last_curr * point_)/dR)  t_last_curr, d与R无关
                 *         = w.dot(-Rpoint_X)   公式 d(Rv)/dR = -RvX X表示反对称
                 *         = -w.transpose() * R * Utility::skewSymmetric(point_)
                 *
                 */
                jaco.rightCols<3>() = -w.transpose() * R * Utility::skewSymmetric(point_);

                jacobian_pose.setZero();
                jacobian_pose.leftCols<6>() = jaco;
            }
        }
        return true;
    }

    // TODO: check if derived jacobian == perturbation on the raw function
    //用来验证当前定义的LidarScanPlaneNormFactor类中Evaluate函数中的解析求导是否正确
    void check(double **param)
    {
        double *res = new double[1];
        double **jaco = new double *[1];
        jaco[0] = new double[1 * 7];
        Evaluate(param, res, jaco);
        std::cout << "[LidarScanPlaneNormFactor] check begins" << std::endl;
        std::cout << "analytical:" << std::endl;

        std::cout << res[0] << std::endl;
        std::cout << Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>>(jaco[0]) << std::endl;

        delete[] jaco[0];
        delete[] jaco;
        delete[] res;

        Eigen::Quaterniond q_last_curr(param[0][6], param[0][3], param[0][4], param[0][5]);
        Eigen::Vector3d t_last_curr(param[0][0], param[0][1], param[0][2]);
        q_last_curr = Eigen::Quaterniond::Identity().slerp(s_, q_last_curr);
        t_last_curr = s_ * t_last_curr;

        Eigen::Vector3d w(coeff_(0), coeff_(1), coeff_(2));
        double d = coeff_(3);
        double a = w.dot(q_last_curr * point_ + t_last_curr) + d;
        double r = a;

        std::cout << "perturbation:" << std::endl;
        std::cout << r << std::endl;

        const double eps = 1e-6;
        Eigen::Matrix<double, 1, 6> num_jacobian;

        // add random perturbation
        //添加随机扰动
        for (int k = 0; k < 6; k++)
        {
            Eigen::Quaterniond q_last_curr(param[0][6], param[0][3], param[0][4], param[0][5]);
            Eigen::Vector3d t_last_curr(param[0][0], param[0][1], param[0][2]);
            q_last_curr = Eigen::Quaterniond::Identity().slerp(s_, q_last_curr);
            t_last_curr = s_ * t_last_curr;

            int a = k / 3, b = k % 3;
            //分别对三个轴加扰动
            Eigen::Vector3d delta = Eigen::Vector3d(b == 0, b == 1, b == 2) * eps;
            //分别对平移量和旋转量加扰动
            if (a == 0)
                t_last_curr += delta;
            else if (a == 1)
                q_last_curr = q_last_curr * Utility::deltaQ(delta);

            Eigen::Vector3d w(coeff_(0), coeff_(1), coeff_(2));
            double d = coeff_(3);
            double v = w.dot(q_last_curr * point_ + t_last_curr) + d;
            double tmp_r = v;
            num_jacobian(0, k) = (tmp_r - r) / eps;
        }
        std::cout << num_jacobian << std::endl;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    Eigen::Vector3d point_;
    Eigen::Vector4d coeff_;
    double s_;
};

// ****************************************************************
// calculate distrance from point to edge
class LidarScanEdgeFactor : public ceres::SizedCostFunction<1, 7>
{
public:
    LidarScanEdgeFactor(const Eigen::Vector3d &point,
                        const Eigen::VectorXd &coeff,
                        const double &s = 1.0)
        : point_(point), coeff_(coeff), s_(s) {}

    bool Evaluate(double const *const *param, double *residuals, double **jacobians) const
    {
        Eigen::Quaterniond q_last_curr(param[0][6], param[0][3], param[0][4], param[0][5]);
        Eigen::Vector3d t_last_curr(param[0][0], param[0][1], param[0][2]);
        q_last_curr = Eigen::Quaterniond::Identity().slerp(s_, q_last_curr);
        t_last_curr = s_ * t_last_curr;

        Eigen::Vector3d lpa(coeff_(0), coeff_(1), coeff_(2));
        Eigen::Vector3d lpb(coeff_(3), coeff_(4), coeff_(5));
        Eigen::Vector3d lp = q_last_curr * point_ + t_last_curr;

        Eigen::Vector3d nu = (lp - lpa).cross(lp - lpb);
        Eigen::Vector3d de = lpa - lpb;
        residuals[0] = nu.norm() / de.norm();

        if (jacobians)
        {
            Eigen::Matrix3d R = q_last_curr.toRotationMatrix();
            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> jacobian_pose(jacobians[0]);
                Eigen::Matrix<double, 1, 6> jaco; // [dy/dt, dy/dq, 1]

                Eigen::Matrix<double, 1, 3> eta = 1.0 / de.norm() * nu.normalized().transpose();
                jaco.leftCols<3>() = -eta * Utility::skewSymmetric(lpa - lpb);
                jaco.rightCols<3>() = eta * Utility::skewSymmetric(lpa - lpb) * R * Utility::skewSymmetric(point_);

                jacobian_pose.setZero();
                jacobian_pose.leftCols<6>() = jaco;
            }
        }
        return true;
    }

    void check(double **param)
    {
        double *res = new double[1];
        double **jaco = new double *[1];
        jaco[0] = new double[1 * 7];
        Evaluate(param, res, jaco);
        std::cout << "[LidarScanEdgeFactor] check begins" << std::endl;
        std::cout << "analytical:" << std::endl;
        std::cout << res[0] << std::endl;
        std::cout << Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>>(jaco[0]) << std::endl;

        delete[] jaco[0];
        delete[] jaco;
        delete[] res;

        Eigen::Quaterniond q_last_curr(param[0][6], param[0][3], param[0][4], param[0][5]);
        Eigen::Vector3d t_last_curr(param[0][0], param[0][1], param[0][2]);

        Eigen::Vector3d lpa(coeff_(0), coeff_(1), coeff_(2));
        Eigen::Vector3d lpb(coeff_(3), coeff_(4), coeff_(5));
        Eigen::Vector3d lp = q_last_curr * point_ + t_last_curr;

        Eigen::Vector3d nu = (lp - lpa).cross(lp - lpb);
        Eigen::Vector3d de = lpa - lpb;
        double r = nu.norm() / de.norm();

        std::cout << "perturbation:" << std::endl;
        std::cout << r << std::endl;

        const double eps = 1e-6;
        Eigen::Matrix<double, 1, 6> num_jacobian;

        // add random perturbation
        for (int k = 0; k < 6; k++)
        {
            Eigen::Quaterniond q_last_curr(param[0][6], param[0][3], param[0][4], param[0][5]);
            Eigen::Vector3d t_last_curr(param[0][0], param[0][1], param[0][2]);
            int a = k / 3, b = k % 3;
            Eigen::Vector3d delta = Eigen::Vector3d(b == 0, b == 1, b == 2) * eps;
            if (a == 0)
                t_last_curr += delta;
            else if (a == 1)
                q_last_curr = q_last_curr * Utility::deltaQ(delta);

            Eigen::Vector3d lpa(coeff_(0), coeff_(1), coeff_(2));
            Eigen::Vector3d lpb(coeff_(3), coeff_(4), coeff_(5));
            Eigen::Vector3d lp = q_last_curr * point_ + t_last_curr;

            Eigen::Vector3d nu = (lp - lpa).cross(lp - lpb);
            Eigen::Vector3d de = lpa - lpb;
            double tmp_r = nu.norm() / de.norm();
            num_jacobian(k) = (tmp_r - r) / eps;
        }
        std::cout << num_jacobian << std::endl;
    }

private:
    const Eigen::Vector3d point_;
    const Eigen::VectorXd coeff_;
    const double s_;
};

// ****************************************************************
// calculate distrance from point to edge (using 3*1 vector)
class LidarScanEdgeFactorVector : public ceres::SizedCostFunction<3, 7>
{
public:
    LidarScanEdgeFactorVector(const Eigen::Vector3d &point,
                              const Eigen::VectorXd &coeff,
                              const double &s = 1.0)
        : point_(point), coeff_(coeff), s_(s) {}

    bool Evaluate(double const *const *param, double *residuals, double **jacobians) const
    {
        Eigen::Quaterniond q_last_curr(param[0][6], param[0][3], param[0][4], param[0][5]);
        Eigen::Vector3d t_last_curr(param[0][0], param[0][1], param[0][2]);
        q_last_curr = Eigen::Quaterniond::Identity().slerp(s_, q_last_curr);
        t_last_curr = s_ * t_last_curr;

        //lpa和lpb为两个近邻点
        Eigen::Vector3d lpa(coeff_(0), coeff_(1), coeff_(2));
        Eigen::Vector3d lpb(coeff_(3), coeff_(4), coeff_(5));
        Eigen::Vector3d lp = q_last_curr * point_ + t_last_curr;

        //lp与lpa组成的向量叉乘lp与lpb组成的向量 表示：lp、lpa、lpb三个点组成的平行四边形的面积 参考:https://zhuanlan.zhihu.com/p/404326817
        Eigen::Vector3d nu = (lp - lpa).cross(lp - lpb);
        //lpa与lpb组成的向量,为三角形的底
        Eigen::Vector3d de = lpa - lpb;
        //计算点到直线的距离，为高
        residuals[0] = nu.x() / de.norm();
        residuals[1] = nu.y() / de.norm();
        residuals[2] = nu.z() / de.norm();
        // residuals[0] = nu.norm / de.norm();

        if (jacobians)
        {
            Eigen::Matrix3d R = q_last_curr.toRotationMatrix();
            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> jacobian_pose(jacobians[0]);
                Eigen::Matrix<double, 3, 6> jaco; // [dy/dt, dy/dR, 1]

                //分母与dt，dR无关，为常量
                double eta = 1.0 / de.norm();
                /*!
                 *  对平移量求导dt  https://blog.csdn.net/weixin_43851636/article/details/125340140?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-125340140-blog-103102216.pc_relevant_multi_platform_whitelistv3&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-125340140-blog-103102216.pc_relevant_multi_platform_whitelistv3&utm_relevant_index=1
                 *  d((lp-lpa)x(lp-lpb))/dt = -skew(lp-lpb)*(d(lp-lpa)/dt) + skew(lp-lpa)*(d(lp-lpb)/dt)  根据向量叉乘求导公式: d(UxV)/dW = -skew(V)*(dU/dW)+skew(U)*(dV/dW)
                 *                          = -skew(lp-lpb)+skew(lp-lpa)   d(lp-lpa)/dt = 1
                 *                          = skew(lpb-lp)+skew(lp-lpa)
                 *                          = skew(lpb-lpa)
                 *                          = -skew(lpa-lpb)
                 */

                jaco.leftCols<3>() = -eta * Utility::skewSymmetric(lpa - lpb);
                /*!
                 *
                 * 对旋转矩阵求导dR
                 * d((lp-lpa)x(lp-lpb))/dR = -skew(lp-lpb)*(d(lp-lpa)/dt) + skew(lp-lpa)*(d(lp-lpb)/dt)
                 *                         = -skew(lp-lpb)*(-R*skew(point_)) + skew(lp-lpa)*(-R*skew(point_))   公式 d(Rv)/dR = -RvX X表示反对称
                 *                         = skew(lp-lpb)*(R*skew(point_))-skew(lp-lpa)*(R*skew(point_))
                 *                         = skew(lpa-lpb)*R*skew(point_)
                 *
                 */
                jaco.rightCols<3>() = eta * Utility::skewSymmetric(lpa - lpb) * R * Utility::skewSymmetric(point_);

                jacobian_pose.setZero();
                jacobian_pose.leftCols<6>() = jaco;
            }
        }
        return true;
    }

    void check(double **param)
    {
        double *res = new double[3];
        double **jaco = new double *[1];
        jaco[0] = new double[3 * 7];
        Evaluate(param, res, jaco);
        std::cout << "[LidarScanEdgeFactor] check begins" << std::endl;
        std::cout << "analytical:" << std::endl;
        std::cout << res[0] << " " << res[1] << " " << res[2] << std::endl;
        std::cout << Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>>(jaco[0]) << std::endl;

        delete[] jaco[0];
        delete[] jaco;
        delete[] res;

        Eigen::Quaterniond q_last_curr(param[0][6], param[0][3], param[0][4], param[0][5]);
        Eigen::Vector3d t_last_curr(param[0][0], param[0][1], param[0][2]);

        Eigen::Vector3d lpa(coeff_(0), coeff_(1), coeff_(2));
        Eigen::Vector3d lpb(coeff_(3), coeff_(4), coeff_(5));
        Eigen::Vector3d lp = q_last_curr * point_ + t_last_curr;

        Eigen::Vector3d nu = (lp - lpa).cross(lp - lpb);
        Eigen::Vector3d de = lpa - lpb;
        Eigen::Vector3d r = nu / de.norm();

        std::cout << "perturbation:" << std::endl;
        std::cout << r.transpose() << std::endl;

        const double eps = 1e-6;
        Eigen::Matrix<double, 3, 6> num_jacobian;

        // add random perturbation
        for (int k = 0; k < 6; k++)
        {
            Eigen::Quaterniond q_last_curr(param[0][6], param[0][3], param[0][4], param[0][5]);
            Eigen::Vector3d t_last_curr(param[0][0], param[0][1], param[0][2]);
            int a = k / 3, b = k % 3;
            Eigen::Vector3d delta = Eigen::Vector3d(b == 0, b == 1, b == 2) * eps;
            if (a == 0)
                t_last_curr += delta;
            else if (a == 1)
                q_last_curr = q_last_curr * Utility::deltaQ(delta);

            Eigen::Vector3d lpa(coeff_(0), coeff_(1), coeff_(2));
            Eigen::Vector3d lpb(coeff_(3), coeff_(4), coeff_(5));
            Eigen::Vector3d lp = q_last_curr * point_ + t_last_curr;

            Eigen::Vector3d nu = (lp - lpa).cross(lp - lpb);
            Eigen::Vector3d de = lpa - lpb;
            Eigen::Vector3d tmp_r = nu / de.norm();
            num_jacobian.col(k) = (tmp_r - r) / eps;
        }
        std::cout << num_jacobian.block<1, 6>(0, 0) << std::endl;
        std::cout << num_jacobian.block<1, 6>(1, 0) << std::endl;
        std::cout << num_jacobian.block<1, 6>(2, 0) << std::endl;
    }

private:
    const Eigen::Vector3d point_;
    const Eigen::VectorXd coeff_;
    const double s_;
};
