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

#include "initial_extrinsics.h"
#include <cstdlib>
#include <ctime>

using namespace Eigen;

const double EPSILON_R = 0.05;
const double EPSILON_T = 0.1;
const size_t N_POSE = 300; //容器内最大POSE数量

InitialExtrinsics::InitialExtrinsics() {}

void InitialExtrinsics::clearState()
{
    calib_ext_.clear();

    cov_rot_state_.clear();
    cov_pos_state_.clear();

    v_rot_cov_.clear();
    v_pos_cov_.clear();

    v_pose_.clear();

    frame_cnt_ = 0;
    pose_cnt_ = 0;

    Q_.clear();
}

/*!
 * @brief 设置初始参数
 */
void InitialExtrinsics::setParameter()
{
    // QBL[i] = Eigen::Quaterniond::Identity();
    // TBL[i] = Eigen::Vector3d::Zero();
    for (size_t i = 0; i < NUM_OF_LASER; i++) calib_ext_.push_back(Pose(QBL[i], TBL[i], TDBL[i]));

    cov_rot_state_ = std::vector<bool>(NUM_OF_LASER, false);
    cov_rot_state_[IDX_REF] = true;
    full_cov_rot_state_ = false;

    cov_pos_state_ = std::vector<bool>(NUM_OF_LASER, false);
    cov_pos_state_[IDX_REF] = true;
    full_cov_pos_state_ = false;

    v_rot_cov_.resize(NUM_OF_LASER);
    v_pos_cov_.resize(NUM_OF_LASER);
    rot_cov_thre_ = (PLANAR_MOVEMENT) ? 0.05 : 0.25;
    printf("[InitialExtrinsics] rot cov thre: %f\n", rot_cov_thre_);

    Q_.resize(NUM_OF_LASER);
    for (size_t i = 0; i < Q_.size(); i++) Q_[i] = Eigen::MatrixXd::Zero(N_POSE * 4, 4);
}

/*!
 * @brief       设置每个雷达的旋转协方差状态
 * @param idx   雷达编号
 * @return
 */
bool InitialExtrinsics::setCovRotation(const size_t &idx)
{
    assert(idx < NUM_OF_LASER);
    cov_rot_state_[idx] = true;
    if (std::find(cov_rot_state_.begin(), cov_rot_state_.end(), false) == cov_rot_state_.end()) full_cov_rot_state_ = true;
}

/*!
 * @brief      设置每个雷达的平移量是否已经完成初始标定
 * @param idx  雷达编号
 * @return
 */
bool InitialExtrinsics::setCovTranslation(const size_t &idx)
{
    assert(idx < NUM_OF_LASER);
    cov_pos_state_[idx] = true;
    if (std::find(cov_pos_state_.begin(), cov_pos_state_.end(), false) == cov_pos_state_.end()) full_cov_pos_state_ = true;
}

/*!
 * @brief 添加帧间相对位姿
 * @param pose_laser 多个激光雷达的位姿
 * @return
 */
bool InitialExtrinsics::addPose(const std::vector<Pose> &pose_laser)
{
    assert(pose_laser.size() == NUM_OF_LASER);
    bool b_check = true;
    for (size_t i = 0; i < pose_laser.size(); i++) b_check = (b_check && checkScrewMotion(pose_laser[IDX_REF], pose_laser[i]));
    if (b_check)
    {
        if (pq_pose_.size() < N_POSE)
        {
            pose_laser_add_ = std::make_pair(pq_pose_.size(), pose_laser);
            v_pose_.push_back(pose_laser_add_.second);
            pq_pose_.push(pose_laser_add_);
        } else
        {
            // maintain a min heap 维护一个堆,消除具有小旋转的约束，如果pose满了，从顶层弹出，加入的pose按w排序后插入
            pose_laser_add_ = std::make_pair(pq_pose_.top().first, pose_laser);
            v_pose_[pose_laser_add_.first] = pose_laser_add_.second;
            pq_pose_.pop();
            pq_pose_.push(pose_laser_add_);
        }        
        // std::cout << pq_pose_.top().second[0].q_.w() << std::endl;
    }
    return b_check;
}

/*!
 * @brief 检查帧间运动是否存在跳变
 * @param pose_ref
 * @param pose_data
 * @return
 */
bool InitialExtrinsics::checkScrewMotion(const Pose &pose_ref, const Pose &pose_data)
{
    AngleAxisd ang_axis_ref(pose_ref.q_);
    AngleAxisd ang_axis_data(pose_data.q_);
    double r_dis = abs(ang_axis_ref.angle() - ang_axis_data.angle());
    double t_dis = abs(pose_ref.t_.dot(ang_axis_ref.axis()) - pose_data.t_.dot(ang_axis_data.axis()));
    // std::cout << "ref pose : " << pose_ref << std::endl;
    // std::cout << "data pose : " << pose_data << std::endl;
    // printf("r_dis: %f, t_dis: %f \n", r_dis, t_dis);
    // v_rd_.push_back(r_dis);
    // v_td_.push_back(t_dis);

    //角度变化与平移变化要小于阈值
    return (r_dis < EPSILON_R) && (t_dis < EPSILON_T);
}

// The quaternion is used in the Hamilton formula
bool InitialExtrinsics::calibExRotation(const size_t &idx_ref, const size_t &idx_data, Pose &calib_result)
{
    assert(idx_ref < NUM_OF_LASER);
    assert(idx_data < NUM_OF_LASER);

    // ------------------------------- initial rotation
    if (pq_pose_.size() < WINDOW_SIZE) return false;
    const size_t indice = pose_laser_add_.first;
    const std::vector<Pose> &pose_laser = pose_laser_add_.second;
    {
        const Pose &pose_ref = pose_laser[idx_ref];
        const Pose &pose_data = pose_laser[idx_data];

        Eigen::Quaterniond r1 = pose_ref.q_;
        //将当前雷达坐标系下的相对位姿转换到参考雷达坐标系下
        Eigen::Quaterniond r2 = calib_ext_[idx_data].q_ * pose_data.q_ * calib_ext_[idx_data].inverse().q_;

        //angularDistance就是计算两个坐标系之间相对旋转矩阵在做轴角变换后(u * theta)
        //的角度theta, theta越小说明两个坐标系的姿态越接近，这个角度距离用于后面计算权重，
        //这里计算权重就是为了降低外点的干扰，意思就是为了防止出现误差非常大的R_bk+1^bk和
        //R_ck+1^ck约束导致估计的结果偏差太大
        //
        //这里权重用来抑制过大残差值的影响
        double angular_distance = 180 / M_PI * r1.angularDistance(r2); // calculate delta_theta=|theta_1-theta_2|
        //huber核函数做加权
        double huber = angular_distance > 5.0 ? 5.0 / angular_distance : 1.0; // the derivative of huber norm

        /*!
         * https://blog.csdn.net/try_again_later/article/details/104918393
         * 机器人手眼标定的方法计算出外参矩阵
         *  lidar0 k-1      lidar0 k-1    lidar0      lidar0     lidar1 k-1
         * R             = R           * R         = R        * R
         *  lidar1 k        lidar0 k      lidar1      lidar1     lidar1 k
         *
         *  用四元数重写：
         *   lidar0 k-1            lidar0      lidar0             lidar1 k-1
         *  q           四元数乘法  q         = q        四元数乘法 q
         *   lidar0 k              lidar1      lidar1             lidar1 k
         *
         *          lidar0            lidar0                              lidar0
         *   [q]_L*q       = [q]_R * q           ===>    ([q]_L - [q]_R)*q        = 0  ===>  求解手眼标定问题 Ax= 0
         *          lidar1            lidar1                              lidar1
         *
         * L为左四元数矩阵                                                R为右四元数矩阵
         *  [q]_L = [q_w*I+[q_v]X    (q_v)^T]                            [q]_R = [q_w*I-[q_v]X   (q_v)^T]
         *          [-q_v             q_w   ]  X表示叉乘/反对称                    [-q_v            q_w   ]  x表示叉乘/反对称
         *
         *        = [ q_w  -q_z   q_y  q_x]                                    = [ q_w   q_z  -q_y  q_x]
         *          [ q_z   q_w  -q_x  q_y]                                      [-q_z   q_w   q_x  q_y]
         *          [-q_y   q_x   q_w  q_z]                                      [ q_y  -q_x   q_w  q_z]
         *          [-q_x  -q_y  -q_z  q_w]                                      [-q_x  -q_y  -q_z  q_w]
         *
         *
         *
         */

        Eigen::Matrix4d L, R; // Q1 and Q2 to represent the quaternion representation
        double w = pose_ref.q_.w();
        Eigen::Vector3d q = pose_ref.q_.vec();
        L.block<3, 3>(0, 0) = w * Eigen::Matrix3d::Identity() + Utility::skewSymmetric(q); // LeftQuatMatrix
        L.block<3, 1>(0, 3) = q;
        L.block<1, 3>(3, 0) = -q.transpose();
        L(3, 3) = w;

        w = pose_data.q_.w();
        q = pose_data.q_.vec();
        R.block<3, 3>(0, 0) = w * Eigen::Matrix3d::Identity() - Utility::skewSymmetric(q);
        R.block<3, 1>(0, 3) = q;
        R.block<1, 3>(3, 0) = -q.transpose();
        R(3, 3) = w;

        Q_[idx_data].block<4, 4>(indice * 4, 0) = huber * (L - R);
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(Q_[idx_data], Eigen::ComputeFullU | Eigen::ComputeFullV);
    //cout << svd.singularValues().transpose() << endl;

    // if (PLANAR_MOVEMENT)
    // {
    //     // estimateRyx
    //     Eigen::Vector4d t1 = svd.matrixV().block<4,1>(0,2);
    //     Eigen::Vector4d t2 = svd.matrixV().block<4,1>(0,3);
    //     // solve constraint for q_yz: xy = -zw
    //     double s[2];
    //     if (!common::solveQuadraticEquation(t1(0) * t1(1) + t1(2) * t1(3),
    //                                         t1(0) * t2(1) + t1(1) * t2(0) + t1(2) * t2(3) + t1(3) * t2(2),
    //                                         t2(0) * t2(1) + t2(2) * t2(3),
    //                                         s[0], s[1]))
    //     {
    //         std::cout << "# ERROR: Quadratic equation cannot be solved due to negative determinant." << std::endl;
    //         return false;
    //     }
    //     std::vector<Eigen::Quaterniond> q_yx;
    //     q_yx.resize(2);
    //     double yaw[2];
    //     for (auto i = 0; i < 2; ++i)
    //     {
    //         double t = s[i] * s[i] * t1.dot(t1) + 2 * s[i] * t1.dot(t2) + t2.dot(t2);
    //         // solve constraint ||q_yx|| = 1
    //         double b = sqrt(1.0 / t);
    //         double a = s[i] * b;
    //         q_yx[i].coeffs() = a * t1 + b * t2; // [w x y z]
    //         Eigen::Vector3d euler_angles = q_yx[i].toRotationMatrix().eulerAngles(2, 1, 0);
    //         yaw[0] = euler_angles(0);
    //     }
    //     if (fabs(yaw[0]) < fabs(yaw[1]))
    //         calib_ext_[idx_data].q_ = q_yx[0];
    //     else
    //         calib_ext_[idx_data].q_ = q_yx[1];
    // } else
    // {
    //     Eigen::Matrix<double, 4, 1> x = svd.matrixV().col(3);
    //     calib_ext_[idx_data].q_ = Eigen::Quaterniond(x);
    // }
    //为何SVD分解出来的V最后一列向量为Ax=0的最优解 https://zhuanlan.zhihu.com/p/436842573
    Eigen::Matrix<double, 4, 1> x = svd.matrixV().col(3); // [w, x, y, z]
    if (x[0] < 0) x = -x; // use the standard quaternion
    calib_ext_[idx_data].q_ = Eigen::Quaterniond(x);

    //奇异值
    Eigen::Vector3d rot_cov = svd.singularValues().tail<3>(); // singular value
    v_rot_cov_[idx_data].push_back(rot_cov(1));
    printf("------------------- pose_cnt:%lu, ref:%d, data:%d, rot_cov:%f\n", pq_pose_.size(), idx_ref, idx_data, rot_cov(1));
    //为什么奇异值大于0.25才认为标定成功？
    //从上面引知道1~m个奇异值中，最小的三个值通过svd.singularValues().tail<3>()取出来了，
    //第二个为倒数第二小的奇异值，当然最小的奇异值接近0（奇异值大于0，且通常前面几个大的奇异值总和能占所有奇异值总和的一大部分），
    //这里给了个阈值，看其是否足够大，每个奇异值大小可以理解为分方向的力度大小，这边用0.25来限定倒数第二小的奇异值，
    //我的理解是判断不为0的奇异解中，最终求解的奇异值是否是只有一个接近于0的，最优解只有一个，如果有多个接近0，都很小的话，
    //那么该最小奇异解很可能不是最优，从而间接可以判断在标定过程中是否有充足的旋转等。
    if (rot_cov(1) > rot_cov_thre_) // converage, the second smallest sigular value
    {
        calib_result = calib_ext_[idx_data];
        setCovRotation(idx_data);
        // std::ofstream fout(std::string(OUTPUT_FOLDER + "/others/initialization_rotation.txt").c_str(), std::ios::out);
        // fout.precision(8);
        // auto tmp_pq_pose = pq_pose_;
        // while (!tmp_pq_pose.empty())
        // {
        //     auto pose_laser = tmp_pq_pose.top().second;
        //     fout << pose_laser[idx_ref].q_.w() << " "
        //          << pose_laser[idx_ref].q_.x() << " "
        //          << pose_laser[idx_ref].q_.y() << " "
        //          << pose_laser[idx_ref].q_.z() << " "
        //          << pose_laser[idx_data].q_.w() << " "
        //          << pose_laser[idx_data].q_.x() << " "
        //          << pose_laser[idx_data].q_.y() << " "
        //          << pose_laser[idx_data].q_.z() << std::endl;
        //     tmp_pq_pose.pop();
        // }
        // fout.close();

        // fout.open(std::string(OUTPUT_FOLDER + "/others/initialization_Q.txt").c_str(), std::ios::out);
        // fout.precision(8);
        // for (size_t i = 0; i < Q_[idx_data].rows(); i++)
        // {
        //     for (size_t j = 0; j < Q_[idx_data].cols(); j++)
        //     {
        //         fout << Q_[idx_data](i, j) << " ";
        //     }
        //     fout << std::endl;
        // }
        // fout.close(); 
        // std::cout << "saving initialization state" << std::endl;    
        return true;
    }
    else
    {
        return false;
    }
}

bool InitialExtrinsics::calibExTranslation(const size_t &idx_ref, const size_t &idx_data, Pose &calib_result)
{
    assert(idx_ref < NUM_OF_LASER);
    assert(idx_data < NUM_OF_LASER);
    if (((PLANAR_MOVEMENT) && (calibExTranslationPlanar(idx_ref, idx_data))) ||
       ((!PLANAR_MOVEMENT) && (calibExTranslationNonPlanar(idx_ref, idx_data))))
    {
        calib_result = calib_ext_[idx_data];
        setCovTranslation(idx_data);
        return true;
    } else
    {
        return false;
    }
}

bool InitialExtrinsics::calibExTranslationNonPlanar(const size_t &idx_ref, const size_t &idx_data)
{
    const Eigen::Quaterniond &q_zyx = calib_ext_[idx_data].q_;
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(v_pose_.size() * 3, 3);
    Eigen::MatrixXd b = Eigen::MatrixXd::Zero(v_pose_.size() * 3, 1);
    for (size_t i = 0; i < v_pose_.size(); i++)
    {
        const Pose &pose_ref = v_pose_[i][idx_ref];
        const Pose &pose_data = v_pose_[i][idx_data];
        AngleAxisd ang_axis_ref(pose_ref.q_);
        AngleAxisd ang_axis_data(pose_data.q_);
        double t_dis = abs(pose_ref.t_.dot(ang_axis_ref.axis()) - pose_data.t_.dot(ang_axis_data.axis()));
        double huber = t_dis > 0.04 ? 0.04 / t_dis : 1.0;
        A.block<3, 3>(i * 3, 0) = huber * (pose_ref.q_.toRotationMatrix() - Eigen::Matrix3d::Identity());
        b.block<3, 1>(i * 3, 0) = q_zyx * pose_data.t_ - pose_ref.t_;
    }
    Eigen::Vector3d x;
    x = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
    calib_ext_[idx_data] = Pose(q_zyx, x);
    return true;
}

bool InitialExtrinsics::calibExTranslationPlanar(const size_t &idx_ref, const size_t &idx_data)
{
    /*!   lidar0 k-1  lidar0     lidar0  lidar1 k-1
     *   T           T        = T       T
     *    lidar0 k    lidar1     lidar1  lidar1 k
     *
     * 用旋转矩阵与平移向量表示：
     *   [    lidar0 k-1   lidar0 k-1 ]     [    lidar0   lidar0 ]    [    lidar0   lidar0 ]     [    lidar1 k-1   lidar1 k-1 ]
     *   [   R            t           ]     [   R        t       ]    [   R        t       ]     [   R            t           ]
     *   [    lidar0 k     lidar0 k   ]     [    lidar1   lidar1 ] =  [    lidar1   lidar1 ]     [    lidar1 k     lidar1 k   ]
     *   [                            ]     [                    ]    [                    ]     [                            ]
     *   [     0             1        ]     [     0          1   ]    [     0          1   ]     [     0                1     ]
     *
     *
     *   矩阵相乘，计算一行四列
     *    lidar0 k-1   lidar0     lidar0 k-1     lidar0  lidar1 k-1    lidar0
     *   R            t        + t            = R       t           + t
     *    lidar0 k     lidar1     lidar0 k       lidar1  lidar1 k      lidar1
     *
     *   移项得：
     *     lidar0 k-1           lidar0     lidar0  lidar1 k-1    lidar0 k-1
     *  ( R           - I    ) t        = R       t           - t
     *     lidar0 k      3x3    lidar1     lidar1  lidar1 k      lidar0 k
     *
     *  由于平面运动，t_z = 0, roll = 0, pitch = 0, 待估参数 t_x,t_y, yaw:γ
     *
     *          lidar0 k-1              [ t11 ]    lidar0  lidar1 k-1                 lidar0 k-1
     * 令 t2 = t           的前两行  t1 = [ t12 ] = R       t           的前两行   R1 = R           - I    的2X2 upper-left submatrix
     *          lidar0 k                           lidar1  lidar1 k                   lidar0 k      3x3
     *
     *   [  lidar0 k-1          ] [ tx ]   [ cosγ -sinγ][ t11 ]
     *   [ R             - I    ] [    ] = [           ][     ] - t2
     *   [  lidar0 k 2x2    2x2 ] [ ty ]   [ sinγ  cosγ][ t12 ]
     *
     *        [tx]   [ cosγ -sinγ][t11]
     * ==> R1 [ty] - [ sinγ  cosγ][t12]  = -t2
     *
     *        [ t11 -t12]
     * 令 J = [ t12  t11]      G = [R1 J]
     *                  4x4             2x4
     *
     *           [ tx  ]
     * ==> [R1 J][ ty  ] = -t2
     *           [-cosγ]
     *           [-sinγ]
     *
     *   Ax = B 使用SVD求解未知量 tx ty γ
     */


    const Eigen::Quaterniond &q_yx = calib_ext_[idx_data].q_;
    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(v_pose_.size() * 2, 4); // 2Nx4
    Eigen::MatrixXd w = Eigen::MatrixXd::Zero(v_pose_.size() * 2, 1);
    for (size_t i = 0; i < v_pose_.size(); i++)
    {
        const Pose &pose_ref = v_pose_[i][idx_ref];
        const Pose &pose_data = v_pose_[i][idx_data];
        AngleAxisd ang_axis_ref(pose_ref.q_);
        AngleAxisd ang_axis_data(pose_data.q_);
        double t_dis = abs(pose_ref.t_.dot(ang_axis_ref.axis()) - pose_data.t_.dot(ang_axis_data.axis()));
        double huber = t_dis > 0.04 ? 0.04 / t_dis : 1.0;
        Eigen::Matrix2d J = pose_ref.q_.toRotationMatrix().block<2,2>(0, 0) - Eigen::Matrix2d::Identity(); //相当于上面公式的R1

        //计算K的过程等效于 v = q_yx.toRotationMatrix() * pose_data.t_  K << v(0), -v(1), v(1), v(0)
        Eigen::Vector3d n = q_yx.toRotationMatrix().row(2);
        Eigen::Vector3d p = q_yx * (pose_data.t_ - pose_data.t_.dot(n) * n);
        Eigen::Matrix2d K;
        K << p(0), -p(1), p(1), p(0); //相当于上面公式的J
        G.block<2, 2>(i * 2, 0) = huber * J;
        G.block<2, 2>(i * 2, 2) = huber * K;
        w.block<2, 1>(i * 2, 0) = pose_ref.t_.block<2,1>(0,0); //相当于上面公式的t2
    }
    Eigen::Vector4d m = G.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(w);
    Eigen::Vector3d x(-m(0), -m(1), 0);
    double yaw = atan2(m(3), m(2));
    Eigen::Quaterniond q_z(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));
    calib_ext_[idx_data] = Pose(q_z*q_yx, x);
    return true;
}

void InitialExtrinsics::calibTimeDelay(const size_t &idx_ref, const size_t &idx_data)
{

}

void InitialExtrinsics::decomposeE(cv::Mat E,
                                 cv::Mat_<double> &R1, cv::Mat_<double> &R2,
                                 cv::Mat_<double> &t1, cv::Mat_<double> &t2)
{
    cv::SVD svd(E, cv::SVD::MODIFY_A);
    cv::Matx33d W(0, -1, 0,
                  1, 0, 0,
                  0, 0, 1);
    cv::Matx33d Wt(0, 1, 0,
                   -1, 0, 0,
                   0, 0, 1);
    R1 = svd.u * cv::Mat(W) * svd.vt;
    R2 = svd.u * cv::Mat(Wt) * svd.vt;
    t1 = svd.u.col(2);
    t2 = -svd.u.col(2);
}

// save all poses of all lasers in initialization
void InitialExtrinsics::saveStatistics()
{
    try
    {
        if (MLOAM_RESULT_SAVE)
        {
            ofstream fout(std::string(OUTPUT_FOLDER + "initialization.txt").c_str(), ios::out);
            fout.setf(ios::fixed, ios::floatfield);
            fout.precision(5);
            // orientation
            for (size_t i = 0; i < NUM_OF_LASER; i ++)
            {
                fout << 0 << ",";
                for (auto iter = v_pose_[i].begin(); iter != v_pose_[i].end(); iter++)
                {
                    fout << Eigen::AngleAxisd(iter->q_).angle() << ",";
                    if (iter == v_pose_[i].end() - 1) fout << std::endl;
                }
            }
            // rot_cov
            for (size_t i = 0; i < NUM_OF_LASER; i ++)
            {
                fout << 0 << ",";
                for (auto iter = v_rot_cov_[i].begin(); iter != v_rot_cov_[i].end(); iter++)
                {
                    fout << *iter << ",";
                    if (iter == v_rot_cov_[i].end() - 1) fout << std::endl;
                }
            }
            // pos_cov
            for (size_t i = 0; i < NUM_OF_LASER; i ++)
            {
                fout << 0 << ",";
                for (auto iter = v_pos_cov_[i].begin(); iter != v_pos_cov_[i].end(); iter++)
                {
                    fout << *iter << ",";
                    if (iter == v_pos_cov_[i].end() - 1) fout << std::endl;
                }
            }
            fout.close();
        }
    }
    catch (const std::exception &ex)
    {
        std::cout << "Exception was thrown when save initial extrinsics: " << ex.what() << std::endl;
    }
    return;
}
