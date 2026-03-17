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

# 此采集数据的方法能很好的适配没有遥操作设备的机械臂(只能获取joint_state时),直接处理joint state 来制作 action(joint command)

# ================= 机器人配置 =================
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
# 映射数据集特征名到 ROS 话题名
CAMERA_MAPPING = {
    "observation.images.camera_high": "/camera_high/color/image_raw",
    "observation.images.camera_hand": "/camera_hand/color/image_raw"
}
TRAIN_HZ = 20  
RESIZE_W, RESIZE_H = 640, 480

class Ros1ToLeRobotConverter:
    def __init__(self):
        self.bridge = CvBridge()
        self.typestore = get_typestore(Stores.ROS1_NOETIC)
        # 只订阅状态，Action 将由状态偏移产生
        self.topic_map = {
            "observation.state": "/qpos", # 对应你 C++ 发布的当前角度话题
        }
        self.topic_map.update(CAMERA_MAPPING)

    def process_bag(self, bag_path: str):
        data = defaultdict(list)
        target_topics = set(self.topic_map.values())
        topic_to_key = {v: k for k, v in self.topic_map.items()}

        with Reader(bag_path) as reader:
            connections = [c for c in reader.connections if c.topic in target_topics]

            for connection, _, rawdata in reader.messages(connections=connections):
                topic = connection.topic
                feat_key = topic_to_key[topic]
                msg = self.typestore.deserialize_ros1(rawdata, connection.msgtype)

                # 提取 Header Stamp 进行物理对齐
                if hasattr(msg, 'header'):
                    nsecs = getattr(msg.header.stamp, 'nanosec', getattr(msg.header.stamp, 'nsecs', 0))
                    t_sec = msg.header.stamp.sec + nsecs / 1e9
                else:
                    continue
                    
                if "images" in feat_key:
                    try:
                        cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                        rgb_img = cv2.resize(rgb_img, (RESIZE_W, RESIZE_H))
                        data[feat_key].append({"data": rgb_img, "timestamp": t_sec})
                    except Exception as e:
                        print(f"Image Error on topic {topic}: {e}")
                
                elif "observation.state" in feat_key:
                    val = np.array(msg.position, dtype=np.float32)
                    data[feat_key].append({"data": val, "timestamp": t_sec})

        return self.align_and_shift_frames(data)

    def align_and_shift_frames(self, data):
        """
        核心逻辑：对齐数据并利用状态偏移制作 Action
        """
        main_cam_key = "observation.images.camera_high"
        if not data[main_cam_key] or not data["observation.state"]:
            return {}
            
        # 以主相机为基准对齐所有观测 (Images + State)
        main_frames = sorted(data[main_cam_key], key=lambda x: x["timestamp"])
        temp_aligned = defaultdict(list)

        for ref_frame in main_frames:
            t = ref_frame["timestamp"]
            current_sync_sample = {}
            valid_batch = True
            
            for key, frames in data.items():
                if key == main_cam_key:
                    current_sync_sample[key] = ref_frame["data"]
                    continue
                
                f_times = np.array([f["timestamp"] for f in frames])
                idx = np.argmin(np.abs(f_times - t))
                diff = np.abs(f_times[idx] - t)

                # 图像和状态对齐阈值
                threshold = 0.03 if "images" in key else 0.01
                if diff < threshold:
                    current_sync_sample[key] = frames[idx]["data"]
                else:
                    valid_batch = False
                    break
            
            if valid_batch:
                for key, val in current_sync_sample.items():
                    temp_aligned[key].append(val)

        # 制作 Action (Offset 逻辑)
        # 我们定义：t 时刻的 Action = t+1 时刻的 State
        states = temp_aligned["observation.state"]
        if len(states) < 2:
            print("Warning: Episode too short to create actions.")
            return {}

        # Action 是 State 从 index 1 开始的部分
        aligned_final = defaultdict(list)
        aligned_final["action"] = states[1:]
        
        # Observation (State 和 Images) 需要去掉最后一帧，以匹配 Action 的长度
        for key in temp_aligned.keys():
            aligned_final[key] = temp_aligned[key][:-1]
        
        # 验证对齐后长度是否一致
        list_lens = [len(v) for v in aligned_final.values()]
        if len(set(list_lens)) > 1:
            print(f"Error: Length mismatch after shifting: {list_lens}")
            return {}

        print(f"Successfully aligned and shifted: {len(aligned_final['action'])} frames.")
        return aligned_final

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

    for cam_feat_key in CAMERA_MAPPING.keys():
        features[cam_feat_key] = {
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
        image_writer_processes=4, 
        root=root,
    )
    return dataset

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag_dir", type=str, required=True, help="Path to ROS1 .bag files")
    parser.add_argument("--repo_id", type=str, default="realman_self_collected")
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
            for cam_feat_key in CAMERA_MAPPING.keys():
                frame[cam_feat_key] = aligned_data[cam_feat_key][i]
            
            dataset.add_frame(frame)

        dataset.save_episode(task="Self-recorded move task")

    dataset.consolidate()
    print("Dataset consolidated successfully!")

if __name__ == "__main__":
    main()