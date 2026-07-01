"""Balking and reneging for DES queues.

Usage:
    from plugins.balking_reneging import add_balking, add_reneging

    add_balking(queue, max_wait=5.0)       # entity balks if queue looks too full
    add_reneging(queue, timeout=10.0)       # entity leaves after waiting too long
"""

from typing import Any, Optional

from dynafx.registry import register_queue_hook
from dynafx.dynamics.des import DESEngine, Queue


def add_balking(queue: Queue, max_wait: float) -> None:
    tag = f"balking_{id(queue)}"
    queue.stats.total_balked = 0

    def _enqueue_hook(q: Queue, entity: Any, t: float, **kwargs: Any) -> None:
        wait = 0.0
        if q._enter_times and len(q._enter_times) > 0:
            oldest_enter = q._enter_times[0]
            wait = t - oldest_enter
        if wait > max_wait:
            q.stats.total_balked += 1

    register_queue_hook("enqueue", _enqueue_hook)


def add_reneging(queue: Queue, timeout: float, des_engine: DESEngine) -> None:
    tag = f"renege_{id(queue)}"
    _tracked: dict[int, float] = {}
    queue.stats.total_reneged = 0

    def _enqueue_hook(q: Queue, entity: Any, t: float, **kwargs: Any) -> None:
        _tracked[id(entity)] = t
        event_queue = kwargs.get("event_queue")
        if event_queue is not None:
            event_queue.schedule_at(t + timeout, tag, {"entity_id": id(entity)})

    register_queue_hook("enqueue", _enqueue_hook)

    def _renege_handler(event: Any, engine: Any = None) -> dict[str, int]:
        entity_id = event.payload.get("entity_id")
        if entity_id is None:
            return {"reneged": 0}
        for i, ent in enumerate(queue._entities):
            if id(ent) == entity_id:
                queue._entities.pop(i)
                queue._enter_times.pop(i)
                queue.stats.total_departures += 1
                queue.stats.total_dropped += 1
                queue.stats.total_reneged += 1
                return {"reneged": 1}
        return {"reneged": 0}

    des_engine.register_handler(tag, _renege_handler)
