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


//https://blog.csdn.net/xiaoma_bk/article/details/124729948

#pragma once

#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include "../utility/utility.h"

// x: [tx ty tz; qx qy qz qw]
class PoseLocalParameterization : public ceres::LocalParameterization
{
    //该函数实现了 ⊞ ( x , Δ )
    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const;
    //计算Jacobian 矩阵： J = D_2 ⊞ ( x , 0 ) 以行的形式存储
    virtual bool ComputeJacobian(const double *x, double *jacobian) const;
    //参数块所在的环境空间的维度:四元数4+平移量3
    virtual int GlobalSize() const { return 7; };
    //切空间维度的大小:实际是6个自由度，3旋转+3平移量
    virtual int LocalSize() const { return 6; };
public:
    void setParameter();

    bool is_degenerate_;//判断是否退化
    Eigen::Matrix<double, 6, 6> V_update_;
};
