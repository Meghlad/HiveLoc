"""Turn an operator instruction into a MissionPlan on /mission_plan (via the
Layer-3 planner), and print every /plan_decision the supervisor node returns.
This closes the ROS loop: instruction -> plan -> supervisor verdict, all on
topics."""

import sys
import numpy as np
import rclpy
from rclpy.node import Node
from swarm_msgs.msg import MissionPlan, Assignment, PlanDecision

sys.path.append("/workspace/coop-swarm")   # where layer3_vlm_planner lives in the image
import layer3_vlm_planner as planner


class PlanPublisher(Node):
    def __init__(self):
        super().__init__("plan_publisher")
        self.declare_parameter("instruction", "form a tight circle in the center")
        self.declare_parameter("npz", "layer2_isam2_results.npz")
        self.declare_parameter("frame", 100)
        self.pub = self.create_publisher(MissionPlan, "mission_plan", 10)
        self.create_subscription(PlanDecision, "plan_decision", self.on_decision, 10)

        d = np.load(self.get_parameter("npz").value)
        f = self.get_parameter("frame").value
        pos, cov = d["online_r055"][f], d["cov_r055"][f]
        instruction = self.get_parameter("instruction").value

        plan, source = planner.make_plan(instruction, pos, cov)
        self.get_logger().info(f"planner source: {source}; publishing plan "
                               f"'{plan['plan_id']}' with {len(plan['assignments'])} assignments")
        msg = MissionPlan()
        msg.plan_id = plan["plan_id"]
        msg.issued_unix_ms = plan["issued_unix_ms"]
        msg.min_spacing_m = plan["min_spacing_m"]
        msg.assignments = [
            Assignment(vehicle=a["vehicle"],
                       waypoint_north=a["waypoint_ne"][0],
                       waypoint_east=a["waypoint_ne"][1])
            for a in plan["assignments"]]
        # publish a few times so the late-joining supervisor is sure to see it
        self.timer = self.create_timer(0.5, lambda: self.pub.publish(msg))

    def on_decision(self, msg: PlanDecision):
        verdict = "ACCEPTED" if msg.accepted else "REJECTED"
        self.get_logger().info(f"/plan_decision: plan '{msg.plan_id}' {verdict}"
                               + ("" if msg.accepted
                                  else f"  violations={list(msg.violations)}"))


def main():
    rclpy.init()
    rclpy.spin(PlanPublisher())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
