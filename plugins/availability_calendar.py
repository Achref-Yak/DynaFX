"""Resource availability calendars for the DES engine.

Provides shift-based availability windows for Resource pools.
Requests outside defined windows are denied.

Usage:
    from plugins.availability_calendar import ResourceCalendar, register

    cal = ResourceCalendar()
    cal.add_window(start_hour=8, end_hour=18, days_of_week=[0,1,2,3,4])
    register(cal, resource)

Also registers a SCHEDULE_CHECK(t, start_hour, end_hour) builtin
function usable in .sysd expressions, returning 1.0 if t falls
within the window and 0.0 otherwise.
"""

from datetime import datetime, timezone
from typing import Any

from dynafx.registry import register_builtin, register_resource_hook
from dynafx.dynamics.des import Resource


class ResourceCalendar:
    """Shift-based availability calendar for a Resource pool."""

    def __init__(self) -> None:
        self._windows: list[dict[str, Any]] = []

    def add_window(
        self,
        start_hour: float,
        end_hour: float,
        days_of_week: list[int] | None = None,
    ) -> None:
        self._windows.append({
            "start_hour": start_hour,
            "end_hour": end_hour,
            "days_of_week": days_of_week if days_of_week is not None else list(range(7)),
        })

    def is_available(self, t: float) -> bool:
        dt = datetime.fromtimestamp(t * 86400, tz=timezone.utc)
        weekday = dt.weekday()
        frac = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
        for w in self._windows:
            if weekday in w["days_of_week"] and w["start_hour"] <= frac < w["end_hour"]:
                return True
        return False

    def _pre_request_hook(self, resource: Resource, **kwargs: Any) -> bool:
        return self.is_available(kwargs.get("t", 0.0))


def register(calendar: ResourceCalendar, resource: Resource) -> None:
    def hook(*args: Any, **kwargs: Any) -> None:
        t = kwargs.get("t", 0.0)
        if not calendar.is_available(t):
            raise RuntimeError(
                f"Resource '{resource.name}' unavailable at t={t}"
            )
    register_resource_hook("pre_request", hook)


def _schedule_check(t: float, start_hour: float, end_hour: float) -> float:
    dt = datetime.fromtimestamp(t * 86400, tz=timezone.utc)
    frac = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    return 1.0 if start_hour <= frac < end_hour else 0.0


register_builtin("SCHEDULE_CHECK", _schedule_check)
