"""Subscribes: /odom (nav_msgs/Odometry), /path (nav_msgs/Path).
Publishes: /cmd (geometry_msgs/Twist): linear.x = throttle(+)/brake(-) in [-1, 1], angular.z = steer in [-1, 1].
"""
import math

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node

from .pid import PID
from .pure_pursuit import PurePursuit


class ControllerNode(Node):
    def __init__(self):
        super().__init__("controller")
        self.declare_parameter("target_speed", 8.0)
        self.declare_parameter("wheelbase", 2.9)
        self.declare_parameter("rate", 20.0)
        self.pp = PurePursuit(wheelbase=self.get_parameter("wheelbase").value)
        self.pid = PID(0.8, 0.05, 0.0)
        self.path, self.odom = [], None
        self.create_subscription(Path, "path", self.on_path, 1)
        self.create_subscription(Odometry, "odom", self.on_odom, 1)
        self.pub = self.create_publisher(Twist, "cmd", 1)
        self.dt = 1.0 / self.get_parameter("rate").value
        self.create_timer(self.dt, self.tick)

    def on_path(self, msg):
        self.path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]

    def on_odom(self, msg):
        self.odom = msg

    def tick(self):
        if self.odom is None or not self.path:
            return
        p, q, v = self.odom.pose.pose.position, self.odom.pose.pose.orientation, self.odom.twist.twist.linear
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        speed = math.hypot(v.x, v.y)
        steer, _ = self.pp.steer(p.x, p.y, yaw, speed, self.path)
        u = self.pid.step(self.get_parameter("target_speed").value - speed, self.dt)
        cmd = Twist()
        cmd.linear.x, cmd.angular.z = u, steer
        self.pub.publish(cmd)


def main():
    rclpy.init()
    rclpy.spin(ControllerNode())
