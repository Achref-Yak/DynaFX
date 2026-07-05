"""Plugin registry — extension points for custom builtins and DES hooks.

Usage:
    from dynafx.registry import register_builtin, register_resource_hook

    register_builtin("MY_FUNC", lambda x: x * 2)
    register_resource_hook("pre_request", my_handler)
"""

from collections.abc import Callable

_REGISTERED_BUILTINS: dict[str, Callable] = {}
_REGISTERED_RESOURCE_HOOKS: dict[str, list[Callable]] = {}
_REGISTERED_QUEUE_HOOKS: dict[str, list[Callable]] = {}


def register_builtin(name: str, func: Callable) -> None:
    """Register a custom expression function for use in .sysd equations.

    The function becomes available in all expression evaluation contexts
    (stock flows, aux variables, routing conditions, etc.).
    """
    _REGISTERED_BUILTINS[name] = func


def register_resource_hook(event: str, handler: Callable) -> None:
    """Register a hook on Resource actions.

    Events:
        "pre_request":  handler(resource, t, wait, priority, quantity)
        "post_request": handler(resource, t, wait, priority, quantity, granted)
        "pre_release":  handler(resource, t)
        "post_release": handler(resource, t, released)
    """
    _REGISTERED_RESOURCE_HOOKS.setdefault(event, []).append(handler)


def register_queue_hook(event: str, handler: Callable) -> None:
    """Register a hook on Queue actions.

    Events:
        "enqueue": handler(queue, entity, t, event_queue, success)
        "dequeue": handler(queue, entity, t)
        "route":   handler(queue, entity, t, target_queue, state)
    """
    _REGISTERED_QUEUE_HOOKS.setdefault(event, []).append(handler)


def get_registered_builtins() -> dict[str, Callable]:
    return dict(_REGISTERED_BUILTINS)


def get_resource_hooks(event: str) -> list[Callable]:
    return list(_REGISTERED_RESOURCE_HOOKS.get(event, []))


def get_queue_hooks(event: str) -> list[Callable]:
    return list(_REGISTERED_QUEUE_HOOKS.get(event, []))


def clear_all() -> None:
    """Clear all registrations (useful in tests)."""
    _REGISTERED_BUILTINS.clear()
    _REGISTERED_RESOURCE_HOOKS.clear()
    _REGISTERED_QUEUE_HOOKS.clear()
