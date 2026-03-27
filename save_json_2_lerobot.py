# 伪代码,保存json文件和上传hf的逻辑
import os
import time
import json
import cv2
import numpy as np
from pathlib import Path

# 配置参数
TRAIN_HZ = 30  # 目标采集频率
SAVE_INTERVAL = 1.0 / TRAIN_HZ
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

def record_teleop_episode(episode_id, robot, cameras, root_path):
    episode_dir = Path(root_path) / f"episode_{episode_id}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    
    frames_index = []
    print(f"开始录制... 目标频率: {TRAIN_HZ}Hz")
    
    try:
        while True:
            start_time = time.time()
            
            # 1. 获取硬件数据
            # state: 机器人当前状态, action: 遥操作设备给出的指令
            state = robot.get_joint_positions() 
            action = robot.get_target_command() 
            
            frame_data = {
                "timestamp": start_time,
                "observation.state": state.tolist(),
                "action": action.tolist(),
            }
            
            # 2. 获取并保存所有摄像头的图像
            for cam_name, cam_obj in cameras.items():
                img = cam_obj.get_frame()
                img_name = f"{cam_name}_{start_time}.jpg"
                cv2.imwrite(str(episode_dir / img_name), img)
                frame_data[f"observation.images.{cam_name}"] = img_name
            
            frames_index.append(frame_data)
            
            # 3. 频率控制 (精准对齐 TRAIN_HZ)
            elapsed = time.time() - start_time
            if elapsed < SAVE_INTERVAL:
                time.sleep(SAVE_INTERVAL - elapsed)
            
            if stop_signal_triggered(): break # 触发停止信号（如按键）

    finally:
        # 保存索引 JSON
        with open(episode_dir / "trajectory.json", "w") as f:
            json.dump(frames_index, f)
        print(f"录制结束，共 {len(frames_index)} 帧")



# --------------------------------------------------------
import torch
import shutil
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 这里的配置映射到你的硬件
CAMERA_MAPPING = {"top": "observation.images.top", "wrist": "observation.images.wrist"}
RESIZE_W, RESIZE_H = 640, 480

def create_lerobot_dataset(repo_id, root):
    # --- 引用你提供的特征定义逻辑 ---
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
    for cam_feat_key in CAMERA_MAPPING.values():
        features[cam_feat_key] = {
            "dtype": "video",
            "shape": (3, RESIZE_H, RESIZE_W),
            "names": ["channels", "height", "width"],
        }

    dataset_path = Path(root) / repo_id
    if dataset_path.exists():
        shutil.rmtree(dataset_path)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=TRAIN_HZ,
        robot_type="realman",
        features=features,
        use_videos=True,
        image_writer_processes=2, # 双进程对视频进行高效 H264/AV1 编码
        root=root,
    )

def convert_json_to_lerobot(raw_dir, repo_id, output_root):
    dataset = create_lerobot_dataset(repo_id, output_root)
    episode_dirs = sorted(Path(raw_dir).glob("episode_*"))

    for ep_dir in episode_dirs:
        with open(ep_dir / "trajectory.json", "r") as f:
            frames = json.load(f)

        for f_data in frames:
            # 构建 LeRobot 帧
            frame = {
                "observation.state": torch.tensor(f_data["observation.state"]),
                "action": torch.tensor(f_data["action"]),
            }
            # 加载并缩放图片
            for cam_short, cam_feat_key in CAMERA_MAPPING.items():
                img_path = ep_dir / f_data[cam_feat_key]
                frame[cam_feat_key] = Image.open(img_path).resize((RESIZE_W, RESIZE_H))
            
            dataset.add_frame(frame)

        dataset.save_episode(task="Physical robot teleoperation")

    # 整合数据并计算统计信息
    dataset.consolidate()
    # 一键上传到 Hugging Face
    dataset.push_to_hub()
    print("上传完成！")