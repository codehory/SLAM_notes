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

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>
#include <cmath>
#include <cfloat>
#include <map>

#include <eigen3/Eigen/Dense>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "common/types/type.h"
#include "common/algos/math.hpp"

#include "../estimator/parameters.h"
#include "../utility/tic_toc.h"

class ImageSegmenter {
public:
    ImageSegmenter() {
        // printf("%d, %d, %d, %d\n", min_cluster_size_, min_line_size_, segment_valid_point_num_, segment_valid_line_num_);
        std::pair <int8_t, int8_t> neighbor;
        neighbor.first = -1;
        neighbor.second = 0;
        neighbor_iterator_.push_back(neighbor);
        neighbor.first = 0;
        neighbor.second = 1;
        neighbor_iterator_.push_back(neighbor);
        neighbor.first = 0;
        neighbor.second = -1;
        neighbor_iterator_.push_back(neighbor);
        neighbor.first = 1;
        neighbor.second = 0;
        neighbor_iterator_.push_back(neighbor);
    }

    void setParameter(const int &vertical_scans,
                      const int &horizon_scans,
                      const int &min_cluster_size,
                      const int &segment_valid_point_num,
                      const int &segment_valid_line_num);

    template<typename PointType>
    void projectCloud(const typename pcl::PointCloud<PointType> &laser_cloud_in,
                      typename pcl::PointCloud<PointType> &cloud_matrix,
                      Eigen::MatrixXf &range_mat,
                      std::vector <pcl::PointCloud<PointType>> &cloud_scan,
                      std::vector<int> &cloud_scan_order);

    template<typename PointType>
    void segmentCloud(const typename pcl::PointCloud<PointType> &laser_cloud_in,
                      typename pcl::PointCloud<PointType> &laser_cloud_out,
                      typename pcl::PointCloud<PointType> &laser_cloud_outlier,
                      ScanInfo &scan_info);

private:
    int vertical_scans_, horizon_scans_; // vertical_scans_: 竖直scan  horizon_scans: 水平scan，360度/水平方向上角分辨率
    int ground_scan_id_;  //预设打在地面的scan数量
    int min_cluster_size_, segment_valid_point_num_, segment_valid_line_num_;
    float ang_res_x_, ang_res_y_, ang_bottom_; //ang_res_x_: 单条线束的水平分辨率  ang_res_y_: 线束之间的垂直分辨率
    float segment_alphax_, segment_alphay_; //水平、垂直角分辨率
    std::vector <pair<int8_t, int8_t>> neighbor_iterator_;
};

/*!
 * @brief 将点云投影成图片的方式，线id作为该像素点的行，通过扫描的角分辨率去确定该像素点的列
 * @tparam PointType
 * @param laser_cloud_in
 * @param cloud_matrix      根据图像坐标顺序按行排列的点云
 * @param range_mat         点云投影的图像，像素值为距离
 * @param cloud_scan        每个线束包含的点的集合
 * @param cloud_scan_order  每个线束包含点的个数的集合
 */
// TODO: this part rearrange the point order, which will influence the feature extraction
// project point cloud onto a range image and rearrange the order
template<typename PointType>
void ImageSegmenter::projectCloud(const typename pcl::PointCloud<PointType> &laser_cloud_in,
                                  typename pcl::PointCloud<PointType> &cloud_matrix,
                                  Eigen::MatrixXf &range_mat,
                                  std::vector <pcl::PointCloud<PointType>> &cloud_scan,
                                  std::vector<int> &cloud_scan_order) {
    // convert point cloud to a range image
    float vertical_angle, horizon_angle, range;
    int row_id, column_id;
    for (size_t i = 0; i < laser_cloud_in.size(); i++) {
        PointType point = laser_cloud_in.points[i];
        range = sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
        if (range < ROI_RANGE) continue;

        //计算每个点的俯仰角
        vertical_angle = atan(point.z / sqrt(point.x * point.x + point.y * point.y)) * 180 / M_PI;
        if ((vertical_scans_ == 64) && (ang_res_y_ == FLT_MAX)) // VLP-64
        {
            if (vertical_angle >= -8.83)
                row_id = static_cast<int>((2 - vertical_angle) * 3.0 + 0.5);
            else
                row_id = static_cast<int>(vertical_scans_ / 2) + static_cast<int>((-8.83 - vertical_angle) * 2.0 + 0.5);
            if (vertical_angle > 2 || vertical_angle < -24.33 || row_id > 50 || row_id < 0)
                continue;
        } else {
            row_id = static_cast<int>((vertical_angle + ang_bottom_) / ang_res_y_);
            if (row_id < 0 || row_id >= vertical_scans_)
                continue;
        }

        //horizonAngle的角度范围为[-180, 180]
        horizon_angle = atan2(point.x, point.y) * 180 / M_PI;


        // 为了保证正前方在深度图的中间, horizonAngle-90 将角度范围变为[-270, 90], 本来是x轴朝前，y轴朝右，z轴朝下，y轴指向0度。-90度是将坐标轴绕z轴顺时针转90度，使得x轴指向0度
        // 除以水平角分辨率，round取负之后将范围变为[3/4H, -1/4H]
        // 再加上1/2H, 使columnIdn的变为[5/4H， 1/4H]
        column_id = -round((horizon_angle - 90.0) / ang_res_x_) + horizon_scans_ / 2;
        if (column_id >= horizon_scans_)
            column_id -= horizon_scans_;
        if (column_id < 0 || column_id >= horizon_scans_)
            continue;
        if (range_mat(row_id, column_id) != FLT_MAX)
            continue;

        point.intensity += row_id;
        int index = column_id + row_id * horizon_scans_;
        cloud_matrix.points[index] = point;
        range_mat(row_id, column_id) = range;

        cloud_scan[row_id].push_back(point); // without changing the point order
        cloud_scan_order[index] = cloud_scan[row_id].size() - 1;
    }
}

/*!
 * @brief Segmentation模块，类似lego-LOAM.给点云生成深度图,剔除噪点，点云按线束排序，点云聚类
 * @tparam PointType
 * @param laser_cloud_in        输入点云
 * @param laser_cloud_out       按线束排序好的剔除噪点后的点云
 * @param laser_cloud_outlier   噪声点云
 * @param scan_info             scan信息，记录每条scan起始+5与结束-5的索引
 */
template<typename PointType>
void ImageSegmenter::segmentCloud(const typename pcl::PointCloud<PointType> &laser_cloud_in,
                                  typename pcl::PointCloud<PointType> &laser_cloud_out,
                                  typename pcl::PointCloud<PointType> &laser_cloud_outlier,
                                  ScanInfo &scan_info) {
    // 一帧点云-->由竖直scan为宽，水平scan为长度构成的深度图mat
    // set specific parameters
    Eigen::MatrixXf range_mat = Eigen::MatrixXf::Constant(vertical_scans_, horizon_scans_, FLT_MAX); // 范围矩阵，大小对应投影图像大小，每个值为点云点的range
    Eigen::MatrixXi label_mat = Eigen::MatrixXi::Zero(vertical_scans_, horizon_scans_); //标签矩阵，大小对应投影图像大小，每个值初始化为0

    pcl::PointCloud <PointType> cloud_matrix; //一帧点云
    cloud_matrix.resize(vertical_scans_ * horizon_scans_);
    std::vector <pcl::PointCloud<PointType>> cloud_scan(vertical_scans_); //按线束分类的点云集合
    std::vector<int> cloud_scan_order(vertical_scans_ * horizon_scans_); //
    projectCloud(laser_cloud_in, cloud_matrix, range_mat, cloud_scan, cloud_scan_order);

    std::vector <uint16_t> all_pushed_indx(vertical_scans_ * horizon_scans_);
    std::vector <uint16_t> all_pushed_indy(vertical_scans_ * horizon_scans_);

    std::vector <uint16_t> queue_indx(vertical_scans_ * horizon_scans_);
    std::vector <uint16_t> queue_indy(vertical_scans_ * horizon_scans_);

    std::vector<int> queue_indx_last_negi(vertical_scans_ * horizon_scans_);
    std::vector<int> queue_indy_last_negi(vertical_scans_ * horizon_scans_);
    std::vector<float> queue_last_dis(vertical_scans_ * horizon_scans_);

    // 无效点的标签设为-1
    // remote FLT_MAX points
    for (size_t i = 0; i < vertical_scans_; i++)
        for (size_t j = 0; j < horizon_scans_; j++)
            if (range_mat(i, j) == FLT_MAX)
                label_mat(i, j) = -1;

    // label ground points
    int label_count = 1;
    size_t lower_ind, upper_ind;
    float vertical_angle;
    float diff_x, diff_y, diff_z;

    if (vertical_scans_ == 64) {
        //预设ground_scan_id_的目的是减少计算量，对于雷达水平安装的情况下，一半线束是朝天打的，可以直接忽略
        for (size_t i = ground_scan_id_; i < vertical_scans_; i++) {
            for (size_t j = 0; j < horizon_scans_; j++) {
                if (range_mat(i, j) == FLT_MAX || range_mat(i + 1, j) == FLT_MAX)
                    continue;
                //当前点的索引
                lower_ind = j + i * horizon_scans_;
                //下一行的索引
                upper_ind = j + (i + 1) * horizon_scans_;
                const PointType &point1 = cloud_matrix.points[lower_ind];
                const PointType &point2 = cloud_matrix.points[upper_ind];
                diff_x = point1.x - point2.x;
                diff_y = point1.y - point2.y;
                diff_z = point1.z - point2.z;
                vertical_angle = atan2(diff_z, sqrt(diff_x * diff_x + diff_y * diff_y)) * 180 / M_PI;
                //若夹角小于10°，则将构成向量的两个深度图像素点标记为地面点
                if (abs(vertical_angle) <= 10) // 10deg
                {
                    label_mat(i, j) = label_count;
                    label_mat(i + 1, j) = label_count;
                }
            }
        }
        //对于线束低的雷达，从0开始遍历
    } else {
        for (size_t i = 0; i < ground_scan_id_; i++) {
            for (size_t j = 0; j < horizon_scans_; j++) {
                if (range_mat(i, j) == FLT_MAX || range_mat(i + 1, j) == FLT_MAX)
                    continue;
                lower_ind = j + i * horizon_scans_;
                upper_ind = j + (i + 1) * horizon_scans_;
                const PointType &point1 = cloud_matrix.points[lower_ind];
                const PointType &point2 = cloud_matrix.points[upper_ind];
                diff_x = point1.x - point2.x;
                diff_y = point1.y - point2.y;
                diff_z = point1.z - point2.z;
                vertical_angle = atan2(diff_z, sqrt(diff_x * diff_x + diff_y * diff_y)) * 180 / M_PI;
                if (abs(vertical_angle) <= 10) // 10deg
                {
                    label_mat(i, j) = label_count;
                    label_mat(i + 1, j) = label_count;
                }
            }
        }
    }
    label_count++;

    // BFS to search nearest neighbors
    for (size_t i = 0; i < vertical_scans_; i++) {
        for (size_t j = 0; j < horizon_scans_; j++) {
            // 对未被标记的点执行BFS, 同时进行聚类  (地面标记为1，无效点为-1)
            if (label_mat(i, j) == 0) {
                int row = i;
                int col = j;

                float d1, d2, alpha, angle, dist;
                int from_indx, from_indy, this_indx, this_indy;
                bool line_count_flag[vertical_scans_] = {false};

                // 使用数组模拟队列
                queue_indx[0] = row;
                queue_indy[0] = col;
                queue_indx_last_negi[0] = 0;
                queue_indy_last_negi[0] = 0;
                queue_last_dis[0] = 0;
                int queue_size = 1;
                int queue_start_ind = 0;
                int queue_end_ind = 1;

                all_pushed_indx[0] = row;
                all_pushed_indy[0] = col;
                int all_pushed_ind_size = 1; // 记录分割点云的点数量

                // find the neighbor connecting clusters in range image, bfs
                while (queue_size > 0) {
                    from_indx = queue_indx[queue_start_ind];
                    from_indy = queue_indy[queue_start_ind];
                    --queue_size;
                    ++queue_start_ind;
                    label_mat(from_indx, from_indy) = label_count;
                    for (auto iter = neighbor_iterator_.begin(); iter != neighbor_iterator_.end(); ++iter) {
                        this_indx = from_indx + iter->first;
                        this_indy = from_indy + iter->second;
                        if (this_indx < 0 || this_indx >= vertical_scans_)
                            continue;
                        if ((vertical_scans_ == 64) && (ang_res_y_ == FLT_MAX)) {
                            if (this_indx <= 32)
                                segment_alphay_ = 0.333 / 180.0 * M_PI;
                            else
                                segment_alphay_ = 0.5 / 180.0 * M_PI;
                        }
                        // 调整横轴边界数据是相连的
                        if (this_indy < 0)
                            this_indy = horizon_scans_ - 1;
                        if (this_indy >= horizon_scans_)
                            this_indy = 0;
                        if (label_mat(this_indx, this_indy) != 0)
                            continue;

                        //激光雷达原点到该点的距离AO(取当前点和邻域点的最大值)
                        d1 = std::max(range_mat(from_indx, from_indy),
                                      range_mat(this_indx, this_indy));
                        //激光雷达原点到该点的距离BO(取当前点和邻域点的最小值)
                        d2 = std::min(range_mat(from_indx, from_indy),
                                      range_mat(this_indx, this_indy));
                        //计算两个点的距离AB
                        dist = sqrt(d1 * d1 + d2 * d2 - 2 * d1 * d2 * cos(alpha));
                        //角分辨率 因为找的是上下左右四邻域点，上下使用垂直角分辨率，左右使用水平角分辨率
                        alpha = iter->first == 0 ? segment_alphax_ : segment_alphay_;
                        //参考： https://zhuanlan.zhihu.com/p/435460616
                        /*   \|
                         *    |A
                         *    |\
                         *    | \
                         *    |  \
                         *    |   \
                         *    |beta\
                         *    |     \
                         *    |      \
                         *  H |_______\B
                         *    |       /\
                         *    |      /  \
                         *    |alpha/
                         *    |    /
                         *    |   /
                         *    |  /
                         *    | /
                         *    |/
                         *    O
                         */

                        //beta = arctan(BH/AH)
                        angle = atan2(d2 * sin(alpha), (d1 - d2 * cos(alpha)));
                        //若计算的角度大于segmentTheta，则认为该点和待分配点是同一类点，并把该点的像素坐标保存，便于下次将该点作为基准点
                        //将该点的label值记为和待分配点一致
                        if (angle > SEGMENT_THETA) {
                            // 把此有效点push入队列中
                            queue_indx[queue_end_ind] = this_indx;
                            queue_indy[queue_end_ind] = this_indy;
                            queue_indx_last_negi[queue_end_ind] = iter->first;
                            queue_indy_last_negi[queue_end_ind] = iter->second;
                            queue_last_dis[queue_end_ind] = dist;
                            queue_size++;
                            queue_end_ind++;

                            // 标记此点与中心点同样的label
                            label_mat(this_indx, this_indy) = label_count;
                            line_count_flag[this_indx] = true;

                            //追踪分割的点云
                            all_pushed_indx[all_pushed_ind_size] = this_indx;
                            all_pushed_indy[all_pushed_ind_size] = this_indy;
                            all_pushed_ind_size++;
                        } else if ((iter->second == 0) &&
                                   (queue_indy_last_negi[queue_start_ind] == 0)) // at the same beam 此点与中心点在同一条扫描线上
                        {
                            float dist_last = queue_last_dis[queue_start_ind];
                            if ((dist_last / dist <= 1.2) && ((dist_last / dist >= 0.8))) // inside a plane 判断为同一平面点，这种情况也算同一类
                            {
                                queue_indx[queue_end_ind] = this_indx;
                                queue_indy[queue_end_ind] = this_indy;
                                queue_indx_last_negi[queue_end_ind] = iter->first;
                                queue_indy_last_negi[queue_end_ind] = iter->second;
                                queue_last_dis[queue_end_ind] = dist;
                                queue_size++;
                                queue_end_ind++;

                                label_mat(this_indx, this_indy) = label_count;
                                line_count_flag[this_indx] = true;

                                all_pushed_indx[all_pushed_ind_size] = this_indx;
                                all_pushed_indy[all_pushed_ind_size] = this_indy;
                                all_pushed_ind_size++;
                            }
                        }
                    }
                }

                // check if this segment is valid
                bool feasible_segment = false;
                // 如果点大于30个，认为成功
                if (all_pushed_ind_size >= min_cluster_size_) // cluster_size > min_cluster_size_
                {
                    feasible_segment = true;
                }
                // 如果点大于5个，并且横跨3个纵坐标，认为成功
                else if (all_pushed_ind_size >= segment_valid_point_num_) // line_size > line_mini_size
                {
                    int line_count = 0;
                    for (size_t i = 0; i < vertical_scans_; i++)
                        if (line_count_flag[i])
                            line_count++;

                    if (line_count >= segment_valid_line_num_)
                        feasible_segment = true;
                }

                // 如果为真，则labelCount加1，labelCount代表分割为了多少个簇
                if (feasible_segment) {
                    label_count++;
                } else {
                    // 如果为假，那么标记为999999 判断为噪声点
                    for (size_t i = 0; i < all_pushed_ind_size; ++i) {
                        label_mat(all_pushed_indx[i], all_pushed_indy[i]) = 999999;
                    }
                }
            }
        }
    }

    // filter out outliers from the original point cloud
    laser_cloud_out.clear();
    laser_cloud_outlier.clear();
    if (scan_info.segment_flag_) {
        for (size_t i = 0; i < vertical_scans_; i++) {
            for (size_t j = 0; j < horizon_scans_; j++) {
                if (label_mat(i, j) > 0) // if non-empty points
                {
                    int index = j + i * horizon_scans_;
                    //剔除噪声点
                    if (label_mat(i, j) == 999999) {
                        cloud_scan[i].erase(cloud_scan[i].begin() + cloud_scan_order[index]);
                        // 噪点横轴坐标是5的倍数就进行存储
                        if (j % 5 == 0)
                            laser_cloud_outlier.push_back(cloud_matrix.points[index]);
                    }
                }
            }
        }
    }

    for (size_t i = 0; i < vertical_scans_; i++) {
        //记录每条线束起始与结束索引，跳过前后五个点，用于后面曲率计算
        scan_info.scan_start_ind_[i] = laser_cloud_out.size() + 5;
        laser_cloud_out += cloud_scan[i];
        scan_info.scan_end_ind_[i] = laser_cloud_out.size() - 6;
        // std::cout << i << " " << scan_info.scan_start_ind_[i] << " " << scan_info.scan_end_ind_[i] << std::endl;
    }
    laser_cloud_outlier.push_back(laser_cloud_out.points[0]);
    // std::cout << laser_cloud_out.size() << std::endl;

}

