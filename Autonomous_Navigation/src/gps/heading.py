import torch


class HeadingOffsetEstimator:
    """
    Maintains a heading offset (theta_off) that aligns local-frame odometry
    with global GPS directions.  Offset is updated only after the robot
    has moved a minimum distance, so GPS outliers are ignored.
    All identifiers and comments use plain ASCII.  No gradients are needed,
    but everything stays in torch for consistency and easy .to(device).
    """

    def __init__(self,
                 kp: float = 0.3,
                 ki: float = 0.01,
                 distance_threshold: float = 1.5,
                 device: str = "cpu") -> None:
        """
        kp, ki             : PI gains
        distance_threshold : meters of odometry travel before each update
        device             : "cpu", "cuda", etc.
        """
        self.kp = kp
        self.ki = ki
        self.distance_threshold = distance_threshold
        self.device = device

        self.theta_off = torch.tensor(0.0, device=device)   # radians
        self.integral_err = torch.tensor(0.0, device=device)

        self._odo_ref: torch.Tensor | None = None
        self._gps_ref: torch.Tensor | None = None

    @staticmethod
    def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
        """Wrap angle to the interval [-pi, pi)."""
        return (angle + torch.pi) % (2.0 * torch.pi) - torch.pi

    def update(self,
               time_sec: float,
               odo_pos: torch.Tensor,
               odo_yaw: float,
               gps_pos: torch.Tensor) -> torch.Tensor:
        """
        Call every control tick.

        time_sec : current time (seconds) – kept for logging/future use
        odo_pos  : torch tensor [x, y] in the local frame
        odo_yaw  : local yaw (radians) – not used here, but passed for symmetry
        gps_pos  : torch tensor [E, N] in global ENU frame

        Returns
        -------
        theta_off : torch scalar tensor, radians
            Add this to any high-rate IMU/odometry yaw to get global yaw.
        """
        # Ensure data live on the chosen device
        odo_pos = odo_pos.to(self.device)
        gps_pos = gps_pos.to(self.device)

        # First call: just set reference points
        if self._odo_ref is None or self._gps_ref is None:
            self._odo_ref = odo_pos.clone()
            self._gps_ref = gps_pos.clone()
            return self.theta_off

        # Distance traveled since last update
        dist_traveled = torch.norm(odo_pos - self._odo_ref)
        if dist_traveled < self.distance_threshold:
            return self.theta_off  # not enough motion yet

        # Displacement vectors
        delta_odo = odo_pos - self._odo_ref
        delta_gps = gps_pos - self._gps_ref

        # Rotate odometry displacement by current offset
        c = torch.cos(self.theta_off)
        s = torch.sin(self.theta_off)
        delta_odo_rot = torch.stack([
            c * delta_odo[0] - s * delta_odo[1],
            s * delta_odo[0] + c * delta_odo[1]
        ])

        # Heading error between GPS and rotated odometry
        yaw_gps = torch.atan2(delta_gps[1], delta_gps[0])
        yaw_odo = torch.atan2(delta_odo_rot[1], delta_odo_rot[0])
        yaw_err = self._wrap_angle(yaw_gps - yaw_odo)

        # PI correction
        self.integral_err += yaw_err * dist_traveled
        self.theta_off += self.kp * yaw_err + self.ki * self.integral_err
        self.theta_off = self._wrap_angle(self.theta_off)

        # Reset references for next accumulation window
        self._odo_ref = odo_pos.clone()
        self._gps_ref = gps_pos.clone()

        return torch.rad2deg(self.theta_off)

