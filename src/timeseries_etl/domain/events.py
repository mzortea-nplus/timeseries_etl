"""Alert level computation (EventsController)."""

from typing import Any

import numpy as np


class EventsController:
    """Compute alarm series from z-scores with decay."""

    def __init__(
        self,
        z: np.ndarray,
        time: Any,
        k: float = 3.0,
        decay_rate: float = 0.05,
        alert_th: float = 3.0,
    ) -> None:
        self.z = z
        self.time = time
        self.k = k
        self.decay_rate = decay_rate
        self.alert_th = alert_th

    def run(self) -> tuple[np.ndarray, list[dict]]:
        """Return (alarm_series, alarm_events)."""
        alarm_val = 0.0
        alarm_series: list[float] = []
        alarm_events: list[dict] = []

        for i in range(len(self.z)):
            if alarm_val > 3:
                alarm_val = 0.0
            else:
                p = 1 if self.warning(i) else 0
                alarm_val = alarm_val * (1 - float(self.decay_rate)) + p

            alarm_series.append(alarm_val)

            if self.emergency(alarm_val):
                ts = self.time[i] if not hasattr(self.time, "iloc") else self.time.iloc[i]
                alarm_events.append(
                    {"type": "emergency", "timestamp": ts, "value": alarm_val}
                )

        return np.array(alarm_series), alarm_events

    def warning_above(self, i: int) -> bool:
        return bool(self.z[i] > self.k)

    def warning_below(self, i: int) -> bool:
        return bool(self.z[i] < -self.k)

    def warning(self, i: int) -> bool:
        return self.warning_below(i) or self.warning_above(i)

    def emergency(self, alarm_val: float) -> bool:
        return alarm_val >= self.alert_th
