class PID:
    def __init__(self, kp, ki, kd, out_min=-1.0, out_max=1.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_min, self.out_max = out_min, out_max
        self._i = 0.0
        self._prev = None

    def step(self, error, dt):
        d = 0.0 if self._prev is None else (error - self._prev) / dt
        self._prev = error
        i = self._i + error * dt
        out = self.kp * error + self.ki * i + self.kd * d
        if self.out_min < out < self.out_max:  # anti-windup: only integrate when unsaturated
            self._i = i
        return max(self.out_min, min(self.out_max, out))
