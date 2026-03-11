"""Tests for EventsController."""

import numpy as np

from timeseries_etl.domain.events import EventsController


def test_events_controller_basic():
    z = np.array([0, 0, 4, 0, 0, 0, 0, 0, 0, 0])  # one spike at index 2
    t = np.arange(len(z))
    ctrl = EventsController(z=z, time=t, k=3, decay_rate=0.5, alert_th=3.0)
    alarm_series, events = ctrl.run()
    assert len(alarm_series) == len(z)
    assert alarm_series[2] >= 1
    # decay: alarm_val decreases over time
    assert alarm_series[-1] < alarm_series[2]
