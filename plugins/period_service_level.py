"""Period-based service level tracking.

Usage:
    from plugins.period_service_level import ServiceLevelTracker, register

    tracker = ServiceLevelTracker(period_days=7)
    register(tracker, queue)

    # Later:
    report = tracker.get_service_level(t)  # returns dict with period, fill_rate, total_demand, total_met
"""

from typing import Any

from dynafx.registry import register_queue_hook
from dynafx.dynamics.des import Queue


class ServiceLevelTracker:
    """Tracks fill rate per time window for a DES queue."""

    def __init__(self, period_days: float = 7.0) -> None:
        self.period_days = period_days
        self._period_arrivals: dict[int, int] = {}
        self._period_departures: dict[int, int] = {}
        self._tag = str(id(self))

    def _enqueue_hook(self, queue: Queue, **kwargs: Any) -> None:
        t = kwargs.get("t", 0.0)
        self.record_arrival(t)

    def _dequeue_hook(self, queue: Queue, **kwargs: Any) -> None:
        t = kwargs.get("t", 0.0)
        self.record_departure(t)

    def record_arrival(self, t: float) -> None:
        period = int(t / self.period_days)
        self._period_arrivals[period] = self._period_arrivals.get(period, 0) + 1

    def record_departure(self, t: float) -> None:
        period = int(t / self.period_days)
        self._period_departures[period] = self._period_departures.get(period, 0) + 1

    def get_service_level(self, t: float) -> dict[str, Any]:
        period = int(t / self.period_days)
        demand = self._period_arrivals.get(period, 0)
        met = self._period_departures.get(period, 0)
        return {
            "period": period,
            "period_start": period * self.period_days,
            "period_end": (period + 1) * self.period_days,
            "total_demand": demand,
            "total_met": met,
            "fill_rate": met / demand if demand > 0 else 1.0,
        }


def register(tracker: ServiceLevelTracker, queue: Queue) -> None:
    register_queue_hook("enqueue", tracker._enqueue_hook)
    register_queue_hook("dequeue", tracker._dequeue_hook)
