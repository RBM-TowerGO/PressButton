#!/usr/bin/env python3
"""Render saved sensor evidence without starting Isaac Sim."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Directory produced by verify_astribot_sensors.py")
    args = parser.parse_args()
    snapshot = np.load(args.output / "initial_sensors.npz", allow_pickle=False)
    points = snapshot["lidar_points_local_m"].reshape(-1, 3)[snapshot["lidar_hit"].reshape(-1)]
    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    ax = fig.add_subplot(121, projection="3d")
    scatter = ax.scatter(*points.T, c=points[:, 2], cmap="viridis", s=2)
    ax.scatter([0], [0], [0], c="red", marker="^", s=50, label="Lidar")
    ax.set(xlabel="Local X (m)", ylabel="Local Y (m)", zlabel="Local Z (m)", title="360-degree lidar: valid returns")
    ax.legend()
    fig.colorbar(scatter, ax=ax, shrink=.55, label="Height (m)")
    ax = fig.add_subplot(122)
    ax.scatter(points[:, 0], points[:, 1], c=points[:, 2], cmap="viridis", s=3)
    ax.scatter([0], [0], c="red", marker="^", s=50)
    ax.set(xlabel="Local X (m)", ylabel="Local Y (m)", title="Top view; calibration target at Y = -1 m")
    ax.axis("equal")
    ax.grid(alpha=.2)
    fig.savefig(args.output / "lidar_points.png", dpi=160)
    plt.close(fig)
    data = np.load(args.output / "timeseries.npz", allow_pickle=False)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True, sharex=True)
    for name in ("head", "wrist_left", "wrist_right"):
        axes[0].plot(data["time"], np.linalg.norm(data[f"imu_{name}_ang_vel"], axis=1), label=name)
    axes[0].set(ylabel="Gyro magnitude (rad/s)", title="Measured sensor response in simulation time")
    axes[0].legend()
    axes[1].plot(data["time"], data["contact_astribot_torso_base_force_n"], label="Base contact sensor")
    axes[1].set(ylabel="Contact force (N)")
    axes[1].legend()
    for name in ("astribot_head_joint_1", "astribot_arm_left_joint_7", "astribot_arm_right_joint_7"):
        index = list(data["joint_names"]).index(name)
        axes[2].plot(data["time"], data["q"][:, index], label=name.replace("astribot_", ""))
    axes[2].set(xlabel="Simulation time (s)", ylabel="Joint angle (rad)")
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(args.output / "sensor_response.png", dpi=160)
    plt.close(fig)
    print(args.output / "lidar_points.png")
    print(args.output / "sensor_response.png")


if __name__ == "__main__":
    main()
