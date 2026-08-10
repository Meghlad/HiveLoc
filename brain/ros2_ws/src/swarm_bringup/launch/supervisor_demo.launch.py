"""End-to-end ROS 2 demo of the safety loop:

    estimate_publisher (py) --/swarm_estimate--> supervisor_node (Rust) --/plan_decision--> plan_publisher (py)
    plan_publisher     (py) --/mission_plan---->      ^

Run:  ros2 launch swarm_bringup supervisor_demo.launch.py \
          instruction:="form a tight circle in the center"
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    instruction = LaunchConfiguration("instruction")
    condition = LaunchConfiguration("condition")
    return LaunchDescription([
        DeclareLaunchArgument("instruction",
                              default_value="form a tight circle in the center"),
        DeclareLaunchArgument("condition", default_value="r055"),

        # The safety supervisor (Rust rclrs node) — the only thing that judges plans.
        Node(package="swarm_supervisor_node", executable="supervisor_node",
             name="swarm_supervisor", output="screen"),

        # Estimator replay (pose + marginal covariance).
        Node(package="swarm_bringup", executable="estimate_publisher",
             name="estimate_publisher", output="screen",
             parameters=[{"condition": condition}]),

        # Language-conditioned planner -> /mission_plan; echoes /plan_decision.
        Node(package="swarm_bringup", executable="plan_publisher",
             name="plan_publisher", output="screen",
             parameters=[{"instruction": instruction}]),
    ])
