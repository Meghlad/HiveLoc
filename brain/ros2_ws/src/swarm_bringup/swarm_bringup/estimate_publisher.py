"""Replay the Layer-2 iSAM2 estimate (pose + marginal covariance) onto
/swarm_estimate. In production this node would BE the estimator; here it
replays the saved run so the ROS graph is exercised with real numbers."""

import numpy as np
import rclpy
from rclpy.node import Node
from swarm_msgs.msg import SwarmEstimate


class EstimatePublisher(Node):
    def __init__(self):
        super().__init__("estimate_publisher")
        self.declare_parameter("npz", "layer2_isam2_results.npz")
        self.declare_parameter("condition", "r055")   # healthy radio; r035 = degraded
        self.declare_parameter("rate_hz", 10.0)
        npz = self.get_parameter("npz").value
        cond = self.get_parameter("condition").value
        d = np.load(npz)
        self.pos = d[f"online_{cond}"]                 # [T, n, 2]
        self.cov = d[f"cov_{cond}"]                    # [T, n]
        self.t = 0
        self.pub = self.create_publisher(SwarmEstimate, "swarm_estimate", 10)
        self.timer = self.create_timer(
            1.0 / self.get_parameter("rate_hz").value, self.tick)
        self.get_logger().info(
            f"replaying {self.pos.shape[0]} frames of {cond} estimate -> /swarm_estimate")

    def tick(self):
        t = self.t % self.pos.shape[0]
        msg = SwarmEstimate()
        msg.frame_index = t
        msg.pos_north = self.pos[t, :, 0].astype(float).tolist()
        msg.pos_east = self.pos[t, :, 1].astype(float).tolist()
        msg.cov_trace = self.cov[t].astype(float).tolist()
        self.pub.publish(msg)
        self.t += 1


def main():
    rclpy.init()
    rclpy.spin(EstimatePublisher())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
