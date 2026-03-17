# ROS1 Data Collection & LeRobot Conversion

This project provides a C++ node for real-time robotic data collection and a Python-based converter to transform ROS1 bag files into the **LeRobot** dataset format for imitation learning.

---

## 1. Data Collection

### Run the ROS Package
1. First, launch your robot controller node to handle state publishing and command processing:

```bash
# In your ROS workspace
rosrun your_package_name your_data_collect_node
```

2. In a new terminal, record the required topics. Ensure you capture the joint states, actions, and all relevant camera feeds:

```bash
rosbag record /qpos /action /camera_high/color/image_raw /camera_hand/color/image_raw -O my_tast_data.bag
```

## 2. LeRobot Converter

### Run the Converter

1. Ensure you have the LeRobot installed and configured on your system. You can find the source code and detailed installation guides on the official GitHub page:[https://github.com/huggingface/lerobot]

2. Execute the script to transform your .bag files into a Lerobot training-ready dataset:

```bash
python3 scripts/bag_to_lerobot.py \
    --bag_dir ./path/to/your/bags \
    --repo_id your_username/realman_task_name \
    --root ./output_datasets
```