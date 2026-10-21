import math


class PurePursuit:
    """Steering for a kinematic bicycle model tracking a polyline path."""

    def __init__(self, wheelbase=2.9, k_ld=0.8, min_ld=4.0, max_steer=math.radians(70)):
        self.L, self.k_ld, self.min_ld, self.max_steer = wheelbase, k_ld, min_ld, max_steer

    def steer(self, x, y, yaw, speed, path):
        """path: list of (x, y). Returns (steer in [-1, 1], index of target point)."""
        ld = max(self.min_ld, self.k_ld * speed)
        # nearest point, then first point at least ld further along the path
        i0 = min(range(len(path)), key=lambda i: (path[i][0] - x) ** 2 + (path[i][1] - y) ** 2)
        tgt = i0
        for j in range(i0, i0 + len(path)):
            px, py = path[j % len(path)]
            if math.hypot(px - x, py - y) >= ld:
                tgt = j % len(path)
                break
        px, py = path[tgt]
        alpha = math.atan2(py - y, px - x) - yaw
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))
        delta = math.atan2(2 * self.L * math.sin(alpha), ld)
        return max(-1.0, min(1.0, delta / self.max_steer)), tgt
