r"""
TODO:
This is a kinda nice interface and tool. Move to the general utils package?
"""

from typing import *

import fractions
import numpy as np


class _AlertTracker(object):
    # One time alert storage

    def __init__(self, keys):
        self._keys = set(keys)
        self._store = {k: None for k in self._keys}

    def _set(self, k, v):
        assert k in self._keys
        self._store[k] = v

    def _clear(self):
        for k in self._keys:
            self._store[k] = None

    def __getattr__(self, k):
        if k in self._store:
            v = self._store[k]
            if v is None:
                raise RuntimeError(f'Getting {k} twice before updating')
            self._store[k] = None
            return v
        raise AttributeError()


class StepsCounter(object):
    def __init__(self, *, alert_intervals: Dict[str, int]):
        # alert_intervals is a dict of
        #     alert type => Optional[fractional interval]  (None means never, i.e., inf interval)
        self.alert_intervals = dict(alert_intervals)
        self.reset()

    def reset(self):
        self.alerts = _AlertTracker(self.alert_intervals.keys())
        self._steps = 0
        self._last_alert_steps = {k: -np.inf for k in self.alert_intervals.keys()}
        self._record_alerts()

    def _record_alerts(self):
        # This method records alerts that can later be retrieved via
        # `self.alerts.X` for *just one time*.
        self.alerts._clear()
        for k, interval in self.alert_intervals.items():
            if interval is None:
                self.alerts._set(k, False)
            else:
                alert = (self._steps - self._last_alert_steps[k]) >= interval
                self.alerts._set(k, alert)
                if alert:
                    self._last_alert_steps[k] = self._steps

    def record_alerts_then_update(self, num_steps=1):
        self._record_alerts()
        self._steps += num_steps

    def record_alerts_then_update_to(self, num_total_steps):
        assert num_total_steps >= self._steps
        self._record_alerts()
        self._steps = num_total_steps

    def update_then_record_alerts(self, num_steps=1):
        self._steps += num_steps
        self._record_alerts()

    def update_to_then_record_alerts(self, num_total_steps):
        assert num_total_steps >= self._steps
        self._steps = num_total_steps
        self._record_alerts()
