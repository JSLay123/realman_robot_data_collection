#ifndef _DATA_COLLECT_HPP_
#define _DATA_COLLECT_HPP_

#include <ros/ros.h>
#include <Eigen/Dense>
#include <std_msgs/Bool.h>
#include <std_msgs/Float64MultiArray.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/Twist.h>
#include <tf/tf.h>
#include <sensor_msgs/JointState.h>
#include <thread>

#include "rm_define.h"
#include "rm_interface_global.h"
#include "rm_interface.h"
#include "rm_service.h"

#define tran2rad(x) ((x)*M_PI/180.0) // 角度转弧度
#define tran2deg(x) ((x)*180.0/M_PI) // 弧度转角度

#define mani_dof 6

class ArmController{
public:
    ArmController() : running_(true)
    {
        handle_ = rm_create_robot_arm((char*)"192.168.0.18",8080);
        if (handle_->id == -1)
        {
            ROS_ERROR("Failed to connect to the robot arm!");
            ros::shutdown();
        }
        // 需要手动改成机器人初始ee_pose
        initial_position_ << -0.218, 0.06, 0.357;
        initial_orientation_ << -3.126, 0.001, -0.015;

        // action 和 joint_state发布者
        pub_qpos_ = nh_.advertise<sensor_msgs::JointState>("qpos", 10);
        pub_action_ = nh_.advertise<sensor_msgs::JointState>("action", 10);
        // 订阅来自遥操作设备发来的ee_pose 的增量,注意这里的话题名要对应
        sub_vive_ = nh_.subscribe<geometry_msgs::PoseStamped>("vive_pose", 10, &ArmController::ViveCallback, this);

        // 开一个独立线程：用于采集机器人当前joint_state 并发送出来, 此方法被取代了
        // joint_thread_ = std::thread(&ArmController::jointPublishLoop, this);
    }

    ~ArmController() {
        if (handle_) {
            rm_delete_robot_arm(handle_);
        }
    }

    // 回调函数：处理vive_pose，控制机械臂，以及发布action
    void ViveCallback(const geometry_msgs::PoseStamped::ConstPtr& msg){
        ros::Time sync_time = ros::Time::now();
        float current_degrees[mani_dof];
        int state_res = rm_get_joint_degree(handle_, current_degrees);
        if (state_res != 0) {
            ROS_WARN_THROTTLE(1, "Failed to read joint state, error: %d", state_res);
            return; // 状态读取失败，放弃本帧，确保数据配对的因果性
        }
        // 读取到关节角度，则发布joint_state
        sensor_msgs::JointState qpos_msg;
        qpos_msg.header.stamp = sync_time;
        for (int i = 0; i < mani_dof; i++) {
            qpos_msg.position.push_back(tran2rad(current_degrees[i]));
        }
        // pub_qpos_.publish(qpos_msg);

        // 目标位姿结构体,用于控制机器人
        rm_movep_canfd_mode_t config;
        memset(&config, 0, sizeof(config));
        // 关于这里相加还是相减，正负号自己检测然后需要对齐
        config.pose.position.x = (float)(initial_position_(0) + (-msg->pose.position.x) * 0.9);
        config.pose.position.y = (float)(initial_position_(1) + (-msg->pose.position.y) * 0.9);
        config.pose.position.z = (float)(initial_position_(2) + (msg->pose.position.z) * 0.9);

        // 原数据的四元数直接当作欧拉角进行的传递
        config.pose.euler.rx = (float)(initial_orientation_(0) + msg->pose.orientation.x * 0.8);
        config.pose.euler.ry = (float)(initial_orientation_(1) + msg->pose.orientation.y * 0.8);
        config.pose.euler.rz = (float)(initial_orientation_(2) + msg->pose.orientation.z * 0.8);
        // 设置透传参数
        // config.follow = true;           // 开启高跟随 (1)
        // config.trajectory_mode = 2;     // 滤波模式：处理手部细微抖动，让采集到的 Action 更平滑
        // config.radio = 200;             // 滤波平滑系数 (0-1000)，越大越平滑但延迟稍增

        // 计算目标关节角度action
        rm_inverse_kinematics_params_t ik_params;
        memcpy(ik_params.q_in, current_degrees, sizeof(current_degrees));   // 当前时刻位姿
        ik_params.q_pose = config.pose;     // 目标位姿
        ik_params.flag = 1; // 使用欧拉角进行逆解计算

        float target_degrees[mani_dof]; // 存放输出的角度
        int ik_res = rm_algo_inverse_kinematics(handle_, ik_params, target_degrees);
        
        // 逆解成功，则发布action结果
        if (ik_res == 0){
            sensor_msgs::JointState act_msg ;
            act_msg.header.stamp = sync_time;
            for (int i = 0; i < mani_dof; i++) {
                act_msg.position.push_back(tran2rad(target_degrees[i]));
            }
            pub_action_.publish(act_msg);
            pub_qpos_.publish(qpos_msg);
            
            // 控制机器人的运动
            int result = rm_movep_canfd(handle_, config);
            if (result != 0) {
                // ROS_WARN_THROTTLE 可以防止刷屏
                ROS_WARN_THROTTLE(1, "CANFD Move failed with code: %d", result);
            }
            }
            else{
                ROS_WARN_THROTTLE(1, "Inverse Kinematics Failed, code: %d", ik_res);
                return;
            }
    }

    ros::NodeHandle nh_;
    ros::Subscriber sub_vive_;
    ros::Publisher pub_qpos_;
    ros::Publisher pub_action_;

    rm_robot_handle *handle_;

    Eigen::Vector3d initial_position_;
    Eigen::Vector3d initial_orientation_;

    bool running_;
    // std::thread joint_thread_;
};

#endif