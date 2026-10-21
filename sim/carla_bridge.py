"""CARLA <-> ROS 2 bridge. Owns the simulation clock: each timer tick advances CARLA one step.

Publishes: /odom (nav_msgs/Odometry), /path (nav_msgs/Path, republished at 1 Hz)
Subscribes: /cmd (geometry_msgs/Twist): linear.x = throttle(+)/brake(-), angular.z = steer
Frames: CARLA's native world frame (left-handed), unchanged, so the controller sees the same numbers as run_lap.py.
"""
import math

import carla
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node

from run_lap import build_path, wsl_host

DT = 0.05


class CarlaBridge(Node):
    def __init__(self):
        super().__init__("carla_bridge")
        self.declare_parameter("host", "")
        self.declare_parameter("town", "Town01")
        client = carla.Client(self.get_parameter("host").value or wsl_host(), 2000)
        client.set_timeout(30)
        self.world = client.load_world(self.get_parameter("town").value)
        s = self.world.get_settings()
        s.synchronous_mode, s.fixed_delta_seconds = True, DT
        self.world.apply_settings(s)

        spawn, wps = build_path(self.world)
        self.path_msg = Path()
        self.path_msg.header.frame_id = "map"
        for w in wps:
            ps = PoseStamped()
            ps.pose.position.x, ps.pose.position.y = w.transform.location.x, w.transform.location.y
            self.path_msg.poses.append(ps)
        bp = self.world.get_blueprint_library().find("vehicle.tesla.model3")
        self.car = self.world.spawn_actor(bp, spawn)
        self.cmd = Twist()

        self.create_subscription(Twist, "cmd", lambda m: setattr(self, "cmd", m), 1)
        self.odom_pub = self.create_publisher(Odometry, "odom", 1)
        self.path_pub = self.create_publisher(Path, "path", 1)
        self.create_timer(DT, self.tick)
        self.create_timer(1.0, lambda: self.path_pub.publish(self.path_msg))

    def tick(self):
        u = self.cmd.linear.x
        self.car.apply_control(carla.VehicleControl(
            steer=self.cmd.angular.z, throttle=max(u, 0.0), brake=max(-u, 0.0)))
        self.world.tick()
        t, v = self.car.get_transform(), self.car.get_velocity()
        o = Odometry()
        o.header.stamp = self.get_clock().now().to_msg()
        o.header.frame_id = "map"
        o.pose.pose.position.x, o.pose.pose.position.y = t.location.x, t.location.y
        half = math.radians(t.rotation.yaw) / 2
        o.pose.pose.orientation.z, o.pose.pose.orientation.w = math.sin(half), math.cos(half)
        o.twist.twist.linear.x, o.twist.twist.linear.y = v.x, v.y
        self.odom_pub.publish(o)

    def shutdown(self):
        self.car.destroy()
        s = self.world.get_settings()
        s.synchronous_mode = False
        self.world.apply_settings(s)


def main():
    rclpy.init()
    node = CarlaBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()


if __name__ == "__main__":
    main()
