"""D4.1 — bring up the hive_assist mission stack against a running SITL fleet.

    ANCHOR_INIT -> LOITER_MESH -> TASK_INGEST -> AUCTION -> RECONFIG -> DISPATCH

One `hive_fsm` node drives the FSM; one `task_ingest` node turns an external
detector's geodetic fix into a TacFrame coordinate; one `go_signal` node is the
edge trigger for Domain 3's dispatch. The supervisor is NOT launched here — it
runs natively via run_supervisor.sh, for the reasons in that script.

    ros2 launch hive_bringup mission.launch.py n_vehicles:=20

DDS NOTE. Every vehicle shares one ROS_DOMAIN_ID because the mesh requires them
to see each other; namespaces (/drone_1 ... /drone_20) keep the topics apart.
Anything that does NOT need to see the swarm — monitoring, rosbag replay, a
second experiment — belongs on a different domain ID. At twenty participants the
discovery traffic from a stray stack on the same domain is measurable.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, LogInfo
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node, PushRosNamespace

ARGS = [
    # the surveyed anchor. These are the PX4 SITL default origin, so the sim and
    # the frame agree out of the box; a real site substitutes its survey.
    DeclareLaunchArgument("anchor_lat", default_value="47.397742"),
    DeclareLaunchArgument("anchor_lon", default_value="8.545594"),
    DeclareLaunchArgument("anchor_alt", default_value="488.0"),
    DeclareLaunchArgument(
        "anchor_yaw_deg", default_value="0.0",
        description="CCW angle from ENU East to TacFrame +x, from the survey"),
    DeclareLaunchArgument(
        "anchor_survey_sigma", default_value="0.02",
        description="1-sigma survey accuracy (m). Sizes the anchor factor's "
                    "Sigma_A honestly — do NOT tighten it to make the "
                    "covariance look good, it will hide VIO/range "
                    "inconsistencies instead."),

    DeclareLaunchArgument("n_vehicles", default_value="20"),
    DeclareLaunchArgument("n_slots", default_value="2",
                          description="coalition size for the task"),
    DeclareLaunchArgument("r_comm", default_value="11.0"),
    DeclareLaunchArgument("task", default_value="inspection"),
    DeclareLaunchArgument("standoff_m", default_value="12.0"),

    DeclareLaunchArgument("v_max", default_value="0.9"),
    DeclareLaunchArgument("d_clear", default_value="1.2"),
    DeclareLaunchArgument("horizon", default_value="40",
                          description="ticks of forward simulation per guard"),

    # Domain 1's finding: a single range-only anchor leaves dim ker(H) = 1.
    # Leaving this false means the swarm can rotate freely about the anchor.
    DeclareLaunchArgument(
        "anchor_bearing", default_value="true",
        description="anchor reports bearing in its OWN surveyed frame. Required "
                    "for full rank with one anchor — see hive/nullspace.py. "
                    "Set false ONLY if a second surveyed anchor is configured."),
    DeclareLaunchArgument("second_anchor", default_value="false"),
]


def generate_launch_description() -> LaunchDescription:
    n = LaunchConfiguration("n_vehicles")

    anchor_params = {
        "anchor_lat_deg": LaunchConfiguration("anchor_lat"),
        "anchor_lon_deg": LaunchConfiguration("anchor_lon"),
        "anchor_alt_m": LaunchConfiguration("anchor_alt"),
        "yaw_offset_deg": LaunchConfiguration("anchor_yaw_deg"),
        "survey_sigma_m": LaunchConfiguration("anchor_survey_sigma"),
    }

    estimator = Node(
        package="hive_bringup", executable="anchored_estimator", name="estimator",
        output="screen",
        parameters=[anchor_params, {
            "n_vehicles": n,
            "use_anchor_bearing": LaunchConfiguration("anchor_bearing"),
            "use_second_anchor": LaunchConfiguration("second_anchor"),
        }],
    )

    task_ingest = Node(
        package="hive_bringup", executable="task_ingest", name="task_ingest",
        output="screen",
        parameters=[anchor_params],
        # {lat, lon, alt?, task_id} in, X_tac out. The whole geodetic->metric
        # boundary of this project is this one node.
        remappings=[("~/target_geodetic", "/hive/target_geodetic"),
                    ("~/target_tac", "/hive/target_tac")],
    )

    fsm = Node(
        package="hive_bringup", executable="mission_fsm", name="mission_fsm",
        output="screen",
        parameters=[{
            "n_vehicles": n,
            "n_slots": LaunchConfiguration("n_slots"),
            "r_comm": LaunchConfiguration("r_comm"),
            "v_max": LaunchConfiguration("v_max"),
            "d_clear": LaunchConfiguration("d_clear"),
            "horizon": LaunchConfiguration("horizon"),
            # every emitted stream is compiled from the CURRENT estimate, never
            # resumed from a previous plan. hive/loss_model.py shows why: a
            # stream that keeps advancing while the vehicle is frozen produces a
            # multi-metre lunge the moment the link recovers.
            "replan_from_current_position": True,
        }],
    )

    go_signal = Node(
        package="hive_bringup", executable="go_signal", name="go_signal",
        output="screen",
        parameters=[{
            "task": LaunchConfiguration("task"),
            "standoff_m": LaunchConfiguration("standoff_m"),
        }],
    )

    return LaunchDescription(ARGS + [
        LogInfo(msg=PythonExpression([
            "'hive_assist: ', str(", n, "), ' vehicles, coalition of ', str(",
            LaunchConfiguration("n_slots"), ")"])),
        GroupAction([PushRosNamespace("hive"),
                     estimator, task_ingest, fsm, go_signal]),
    ])
