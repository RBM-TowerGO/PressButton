# Usage on S1B: source scripts/activate_s1b.sh
# towergo Conda Python + the robot's real-hardware ROS environment (Domain 25, Astribot DDS
# profile) + Astribot SDK paths. Unlike /home/astribot/TowerGO/activate.sh, this does not
# switch to the isolated offline Domain 125, because the service drives the real robot.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo 'Run: source scripts/activate_s1b.sh' >&2
    exit 1
fi
PRESSB_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /home/astribot/miniconda3/etc/profile.d/conda.sh
conda activate towergo
source /opt/ros/humble/setup.bash
source /opt/astribot_ros/middle_ware/local_setup.bash
source /opt/astribot_ros/software/local_setup.bash
source /opt/astribot_ros/third_pkg/local_setup.bash
source /opt/astribot_ros/robot_system_ctrl/ros_env.sh
export ASTRIBOT_ROS_INSTALL_PATH=/opt/astribot_ros/software ASTRIBOT_CAMERA_INSTALL_PATH=/opt/astribot_ros/software ASTRIBOT_CAMERA_INFO_PATH=/etc/config/camera_info.json
export ASTRIBOT_SDK_ROOT="${ASTRIBOT_SDK_ROOT:-/home/astribot/astribot_sdk_aarch64}"
export ROBOT_TYPE="${ROBOT_TYPE:-S1}"
export PYTHONPATH="$PRESSB_ROOT/src:$PRESSB_ROOT/scripts:$ASTRIBOT_SDK_ROOT:$ASTRIBOT_SDK_ROOT/third_party/astribot_ros_middleware_py:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$ASTRIBOT_SDK_ROOT/astribot_sdk/core/common/robotics_library_py:$ASTRIBOT_SDK_ROOT/astribot_sdk/core/common/whole_body_control/third_party:$ASTRIBOT_SDK_ROOT/third_party/drake/lib:$ASTRIBOT_SDK_ROOT/third_party/third_pkg"
export PRESSB_ROOT
