使用deepseek按下按钮：
cd /home/astribot/TowerGO/manipulation
source scripts/activate_s1b.sh
python scripts/s1b_agent_run.py --host '' --model-provider deepseek \
  --fingertip-calibration docs/astribot/planar_control/fingertip_calibration/right_wrist_fingertip_point.json \
  --fingertip-state open