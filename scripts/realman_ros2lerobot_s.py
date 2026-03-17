#!/usr/bin/env python3
import os
import shutil
import numpy as np
import torch
import tqdm
import cv2

from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

from pathlib import Path
from collections import defaultdict
from cv_bridge import CvBridge

# LeRobot 相关导入
from lerobot.datasets.lerobot_dataset import LEROBOT_HOME, LeRobotDataset

# 此采集数据的方法能很好的适配没有遥操作设备的机械臂,直接处理joint state 来制作 action(joint command)

# ================= 机器人配置 =================
# 6个机械臂关节 + 1个夹爪
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
CAMERA_TOPICS = ["cam_hand", "cam_top"]     # 修改成具体的相机名字/color/raw,这类话题名
TRAIN_HZ = 20  # 建议采样率
RESIZE_W, RESIZE_H = 640, 480

class Ros1ToLeRobotConverter:
    def __init__(self):
        self.bridge = CvBridge()
        self.typestore = get_typestore(Stores.ROS1_NOETIC)
        # 映射关系：Key(LeRobot特征名) -> Value(ROS1话题名)
        self.topic_map = {
            "observation.state": "/joint_states",
            # "action": "/joint_command"
        }
        for cam in CAMERA_TOPICS:
            self.topic_map[f"observation.images.{cam}"] = f"/{cam}"


    def process_bag(self, bag_path : str):
        data = defaultdict(list)
        target_topics = set(self.topic_map.values())

        with Reader(bag_path) as reader:
            # 建立 topic 到 feat_key 的映射，提高循环效率
            # topic_to_key = {'/joint_states': 'observation.state', ...}
            topic_to_key = {v: k for k, v in self.topic_map.items()}
            # 获取所有连接（话题信息）
            connections = [c for c in reader.connections if c.topic in target_topics]

            for connection, timestamp, rawdata in reader.messages(connections=connections):
                # rosbags 的 timestamp 是纳秒 (ns)，转换为秒 (s)
                t_sec = timestamp / 1e9
                topic = connection.topic
                feat_key = topic_to_key[topic]
                
                # 反序列化消息:将二进制字节流还原为 Python 对象
                msg = self.typestore.deserialize_ros1(rawdata, connection.msgtype)
                if "images" in feat_key:
                    try:
                        # cv_bridge 处理反序列化后的图像对象
                        cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                        rgb_img = cv2.resize(rgb_img, (RESIZE_W, RESIZE_H))
                        data[feat_key].append({"data": rgb_img, "timestamp": t_sec})
                    except Exception as e:
                        print(f"Image Error on topic {topic}: {e}")
                
                elif "observation.state" in feat_key or "action" in feat_key:
                    # 假设是 JointState 类型，提取 position
                    val = np.array(msg.position, dtype=np.float32)
                    data[feat_key].append({"data": val, "timestamp": t_sec})

        return self.align_frames(data)


    def align_frames(self, data):
        """
        以主相机为基准，将不同频率/不同步的话题数据对齐到同一时间线上
        Input:
            data = {
                "observation.images.cam_top": [
                    {"timestamp": 100.0, "data": <img_a1>}, 
                    {"timestamp": 100.1, "data": <img_a2>}, # 每 0.1s 一帧
                ],
                "observation.state": [
                    {"timestamp": 100.02, "data": <state_1>},
                    {"timestamp": 100.05, "data": <state_2>},
                    {"timestamp": 100.08, "data": <state_3>},
                    {"timestamp": 100.11, "data": <state_4>}, # 频率比相机高
                ]
            }
        Output:
            aligned (dict): 严格对齐后的数据字典
                注意：所有 Key 对应的 list 长度现在必须完全相等
                aligned = {
                    "observation.images.cam_top": [
                        <img_a1>, 
                        <img_a2>
                    ],
                    "observation.state": [
                        <state_1>,
                        <state_4>  
                    ],...
                }
        """
        main_cam = f"observation.images.{CAMERA_TOPICS[0]}"
        if not data[main_cam]:
            return {}
            
        timestamps = [f["timestamp"] for f in data[main_cam]]
        aligned = defaultdict(list)

        for t in timestamps:
            valid_frame = True
            current_sync_sample = {}
            
            for key, frames in data.items():
                frame_times = np.array([f["timestamp"] for f in frames])
                idx = np.argmin(np.abs(frame_times - t))
                
                # 10ms 容错
                if np.abs(frame_times[idx] - t) < 0.01:
                    current_sync_sample[key] = frames[idx]["data"]
                else:
                    valid_frame = False
                    break
            
            if valid_frame:
                for key, val in current_sync_sample.items():
                    aligned[key].append(val)
        
        # 将 t+1 时刻的 state 作为 t 时刻的 action
        states = aligned["observation.state"]
        if len(states) < 2:
            return {}

        # Action 是 state 的偏移版本
        # 假设 Action 就是下一时刻的状态
        aligned["action"] = states[1:]
        
        # 为了保持长度一致，必须删掉 (state, img) Key 的最后一帧
        for key in aligned.keys():
            if key != "action":
                aligned[key] = aligned[key][:-1]
            
        assert (len(aligned["action"]) == len(aligned["observation.state"]) and len(aligned["action"]) == len(aligned["observation.image"]) ) , "Length of action, state and image should be the same"
        
        print(f"Aligned {len(aligned.get(main_cam, []))} frames.")
        return aligned

def create_lerobot_dataset(repo_id, root):
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(JOINT_NAMES),),
            "names": {"motors": JOINT_NAMES},
        },
        "action": {
            "dtype": "float32",
            "shape": (len(JOINT_NAMES),),
            "names": {"motors": JOINT_NAMES},
        },
    }

    for cam in CAMERA_TOPICS:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": (3, RESIZE_H, RESIZE_W),
            "names": ["channels", "height", "width"],
        }

    dataset_path = Path(root) / repo_id
    if dataset_path.exists():
        shutil.rmtree(dataset_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=TRAIN_HZ,
        robot_type="realman",
        features=features,
        use_videos=True,
        image_writer_processes=2, # 开启双进程写入视频
        root=root,
    )
    return dataset

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag_dir", type=str, required=True, help="Path to ROS1 .bag files")
    parser.add_argument("--repo_id", type=str, default="jslei/realman_pick_place")
    parser.add_argument("--root", type=str, required=True, help="Output LeRobot dataset path")
    args = parser.parse_args()

    converter = Ros1ToLeRobotConverter()
    dataset = create_lerobot_dataset(args.repo_id, args.root)

    bag_files = sorted(Path(args.bag_dir).glob("*.bag"))
    
    for bag_path in tqdm.tqdm(bag_files, desc="Converting Bags"):
        aligned_data = converter.process_bag(str(bag_path))
        
        if not aligned_data or "action" not in aligned_data:
            continue

        num_frames = len(aligned_data["action"])
        for i in range(num_frames):
            frame = {
                "observation.state": torch.from_numpy(aligned_data["observation.state"][i]),
                "action": torch.from_numpy(aligned_data["action"][i]),
            }
            for cam in CAMERA_TOPICS:
                frame[f"observation.images.{cam}"] = aligned_data[f"observation.images.{cam}"][i]
            
            dataset.add_frame(frame)

        dataset.save_episode(task="Teleop task")

    dataset.consolidate()
    print("Conversion finished!")

if __name__ == "__main__":
    main()