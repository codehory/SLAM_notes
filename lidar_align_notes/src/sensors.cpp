#include "lidar_align/sensors.h"

namespace lidar_align {

    OdomTformData::OdomTformData(Timestamp timestamp_us, Transform T_o0_ot)
            : timestamp_us_(timestamp_us), T_o0_ot_(T_o0_ot) {}

    const Transform &OdomTformData::getTransform() const { return T_o0_ot_; }

    const Timestamp &OdomTformData::getTimestamp() const { return timestamp_us_; }

    void Odom::addTransformData(const Timestamp &timestamp_us, const Transform &T) {
        data_.emplace_back(timestamp_us, T);
    }

    /*!
     * @brief 位姿根据时间戳插值
     * @param timestamp_us 时间戳
     * @param start_idx 起始id
     * @param match_idx 用于下一帧输入的起始id(不重新开始，连接上一帧的id)
     * @return
     */
    Transform Odom::getOdomTransform(const Timestamp timestamp_us,
                                     const size_t start_idx,
                                     size_t *match_idx) const {
        size_t idx = start_idx;

        /// 找到里程计中相邻当前时间戳的前后两帧
        while ((idx < (data_.size() - 1)) &&
               (timestamp_us > data_[idx].getTimestamp())) {
            ++idx;
        }
        if (idx > 0) {
            --idx;
        }

        if (match_idx != nullptr) {
            *match_idx = idx;
        }

        /// 时间戳插值
        // interpolate
        double t_diff_ratio =
                static_cast<double>(timestamp_us - data_[idx].getTimestamp()) /
                static_cast<double>(data_[idx + 1].getTimestamp() -
                                    data_[idx].getTimestamp());

        Transform::Vector6 diff_vector =
                (data_[idx].getTransform().inverse() * data_[idx + 1].getTransform())
                        .log();
        Transform out =
                data_[idx].getTransform() * Transform::exp(t_diff_ratio * diff_vector);

        return out;
    }

    Scan::Scan(const LoaderPointcloud &in, const Config &config)
            : timestamp_us_(in.header.stamp), odom_transform_set_(false) {
        std::default_random_engine generator(in.header.stamp);
        std::uniform_real_distribution<float> distribution(0, 1);

        for (const PointAllFields &point : in) {
            if ((point.intensity > config.min_return_intensity) &&
                distribution(generator) < config.keep_points_ratio) {
                float sq_dist = point.x * point.x + point.y * point.y + point.z * point.z;
                if (std::isfinite(sq_dist) &&
                    (sq_dist > (config.min_point_distance * config.min_point_distance)) &&
                    (sq_dist < (config.max_point_distance * config.max_point_distance))) {
                    Point store_point;
                    store_point.x = point.x;
                    store_point.y = point.y;
                    store_point.z = point.z;
                    store_point.intensity = point.time_offset_us;

                    if (config.estimate_point_times) {
                        // 100000 * 600 / pi
                        const double timing_factor = 19098593.171 / config.lidar_rpm;
                        const double angle = std::atan2(point.x, point.y);

                        // cut out wrap zone
                        if (std::abs(angle) > 3.0) {
                            continue;
                        }
                        store_point.intensity = angle * timing_factor;
                        if (!config.clockwise_lidar) {
                            store_point.intensity *= -1.0;
                        }
                    }
                    raw_points_.push_back(store_point);
                }
            }
        }
        raw_points_.header = in.header;
    }

    Scan::Config Scan::getConfig(ros::NodeHandle *nh) {
        Scan::Config config;
        nh->param("min_point_distance", config.min_point_distance,
                  config.min_point_distance);
        nh->param("max_point_distance", config.max_point_distance,
                  config.max_point_distance);
        nh->param("keep_points_ratio", config.keep_points_ratio,
                  config.keep_points_ratio);
        nh->param("min_return_intensity", config.min_return_intensity,
                  config.min_return_intensity);

        nh->param("estimate_point_times", config.estimate_point_times,
                  config.estimate_point_times);
        nh->param("clockwise_lidar", config.clockwise_lidar, config.clockwise_lidar);
        nh->param("motion_compensation", config.motion_compensation,
                  config.motion_compensation);
        nh->param("lidar_rpm", config.lidar_rpm, config.lidar_rpm);

        return config;
    }

    /*!
     * @brief 将点云帧中的每个点根据时间戳,插值里程计的位姿，得到每个点在里程计坐标系下相对于第一帧的第一个点在里程计坐标系的位姿 T_o0_ot o为里程计坐标系，0为起始时刻，t为t时刻
     * @param odom
     * @param time_offset
     * @param start_idx
     * @param match_idx
     */
    void Scan::setOdomTransform(const Odom &odom, const double time_offset,
                                const size_t start_idx, size_t *match_idx) {
        T_o0_ot_.clear();

        size_t i = 0;
        for (Point point : raw_points_) {
            // NOTE: This static cast is really really important. Without it the
            // timestamp_us will be cast to a float, as it is a very large number it
            // will have quite low precision and when it is cast back to a long int
            // will be a very different value (about 2 to 3 million lower in some
            // quick tests). This difference will then break everything.
            Timestamp point_ts_us = timestamp_us_ +
                                    static_cast<Timestamp>(1000000.0 * time_offset) +
                                    static_cast<Timestamp>(point.intensity);

            T_o0_ot_.push_back(
                    odom.getOdomTransform(point_ts_us, start_idx, match_idx));
        }
        odom_transform_set_ = true;
    }

    const Transform &Scan::getOdomTransform() const {
        if (!odom_transform_set_) {
            throw std::runtime_error(
                    "Attempted to get odom transform before it was set");
        }
        return T_o0_ot_.front();
    }

    /*!
     * @brief 将里程计坐标系下的所有点变换到激光雷达坐标系下, 相当于对点云去畸变的过程
     * @param T_o_l 激光雷达坐标系相对于里程计坐标系的变换
     * @param pointcloud
     */
    void Scan::getTimeAlignedPointcloud(const Transform &T_o_l,
                                        Pointcloud *pointcloud) const {
        for (size_t i = 0; i < raw_points_.size(); ++i) {
            /// T_o_lt = T_o0_ot * T_ot_lt, T_o_l = T_ot_lt刚性连接为固定值
            Transform T_o_lt = T_o0_ot_[i] * T_o_l;

            Eigen::Affine3f pcl_transform;

            pcl_transform.matrix() = T_o_lt.matrix();
            pointcloud->push_back(pcl::transformPoint(raw_points_[i], pcl_transform));
        }
    }

    const Pointcloud &Scan::getRawPointcloud() const { return raw_points_; }

    Lidar::Lidar(const LidarId &lidar_id) : lidar_id_(lidar_id) {};

    const size_t Lidar::getNumberOfScans() const { return scans_.size(); }

    const size_t Lidar::getTotalPoints() const {
        size_t num_points = 0;
        for (const Scan &scan : scans_) {
            num_points += scan.getRawPointcloud().size();
        }
        return num_points;
    }

    const LidarId &Lidar::getId() const { return lidar_id_; }

    void Lidar::addPointcloud(const LoaderPointcloud &pointcloud,
                              const Scan::Config &config) {
        scans_.emplace_back(pointcloud, config);
    }

    /*!
     * @brief 将所有帧点云都统一到起点坐标系下
     * @param pointcloud 是所有帧点云中点的集合
     */
    void Lidar::getCombinedPointcloud(Pointcloud *pointcloud) const {
        for (const Scan &scan : scans_) {
            scan.getTimeAlignedPointcloud(getOdomLidarTransform(), pointcloud);
        }
    }

    void Lidar::saveCombinedPointcloud(const std::string &file_path) const {
        Pointcloud combined;

        getCombinedPointcloud(&combined);
        pcl::PLYWriter writer;
        writer.write(file_path, combined, true);
    }

    /*!
     * @brief 重新设置里程计，如果优化时间偏移，需要重新对里程计插值
     * @param odom
     * @param time_offset
     */
    void Lidar::setOdomOdomTransforms(const Odom &odom, const double time_offset) {
        size_t idx = 0;
        for (Scan &scan : scans_) {
            scan.setOdomTransform(odom, time_offset, idx, &idx);
        }
    }

    void Lidar::setOdomLidarTransform(const Transform &T_o_l) { T_o_l_ = T_o_l; }

    const Transform &Lidar::getOdomLidarTransform() const { return T_o_l_; }

}  // namespace lidar_align
