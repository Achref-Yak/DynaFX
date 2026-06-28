"""Tests for the dynafx plugin registry."""
import pytest
from dynafx.registry import (
    clear_all,
    get_queue_hooks,
    get_registered_builtins,
    get_resource_hooks,
    register_builtin,
    register_queue_hook,
    register_resource_hook,
)


class TestRegistry:
    def teardown_method(self):
        clear_all()

    def test_register_builtin(self):
        register_builtin("DOUBLE", lambda x: x * 2)
        builtins = get_registered_builtins()
        assert "DOUBLE" in builtins
        assert builtins["DOUBLE"](5) == 10

    def test_register_builtin_multiple(self):
        register_builtin("A", lambda: 1)
        register_builtin("B", lambda: 2)
        assert len(get_registered_builtins()) == 2

    def test_resource_hooks_fire(self):
        events = []

        def hook(resource, **kwargs):
            events.append(("pre", resource.name, kwargs.get("t")))

        register_resource_hook("pre_request", hook)
        hooks = get_resource_hooks("pre_request")
        assert len(hooks) == 1

    def test_resource_hooks_empty(self):
        assert get_resource_hooks("nonexistent") == []

    def test_queue_hooks_fire(self):
        events = []

        def hook(queue, **kwargs):
            events.append(("enqueue", queue))

        register_queue_hook("enqueue", hook)
        hooks = get_queue_hooks("enqueue")
        assert len(hooks) == 1

    def test_queue_hooks_empty(self):
        assert get_queue_hooks("nonexistent") == []

    def test_clear_all(self):
        register_builtin("X", lambda: 0)
        register_resource_hook("pre_request", lambda r, **kw: None)
        register_queue_hook("enqueue", lambda q, **kw: None)
        clear_all()
        assert get_registered_builtins() == {}
        assert get_resource_hooks("pre_request") == []
        assert get_queue_hooks("enqueue") == []

    def test_register_builtin_replaces(self):
        register_builtin("F", lambda: 1)
        register_builtin("F", lambda: 2)
        assert get_registered_builtins()["F"]() == 2

    def test_multiple_hooks_for_same_event(self):
        calls = []

        def h1(**kw):
            calls.append(1)

        def h2(**kw):
            calls.append(2)

        register_resource_hook("pre_request", h1)
        register_resource_hook("pre_request", h2)
        hooks = get_resource_hooks("pre_request")
        assert len(hooks) == 2


class TestResourceQuantity:
    """E2: Resource.request(quantity=N)"""

    def teardown_method(self):
        clear_all()

    def test_request_quantity_one_default(self):
        from dynafx.system.des import Resource

        r = Resource("r", capacity=5)
        assert r.request(0.0) is True
        assert r.busy == 1

    def test_request_quantity_multiple(self):
        from dynafx.system.des import Resource

        r = Resource("r", capacity=10)
        assert r.request(0.0, quantity=3) is True
        assert r.busy == 3

    def test_request_quantity_exceeds_capacity(self):
        from dynafx.system.des import Resource

        r = Resource("r", capacity=5)
        assert r.request(0.0, quantity=10) is False
        assert r.busy == 0

    def test_request_quantity_uses_partial_capacity(self):
        from dynafx.system.des import Resource

        r = Resource("r", capacity=5)
        assert r.request(0.0, quantity=3) is True
        assert r.request(0.0, quantity=3) is False
        assert r.busy == 3
        assert r.request(0.0, quantity=2) is True
        assert r.busy == 5

    def test_release_auto_grants_quantity_waiter(self):
        from dynafx.system.des import Resource

        r = Resource("r", capacity=5)
        r.request(0.0, quantity=5)
        assert r.busy == 5
        r.request(0.0, wait=True, quantity=3)
        assert r.waiting == 1
        r.release(1.0)
        # Released 1, waiter needs 3 which exceeds 1 available
        assert r.available == 1
        r.release(2.0)
        assert r.available == 2
        r.release(3.0)
        # Auto-grant: busy was 3 -> release 1 -> busy=2, waiter needs 3 -> 2+3<=5 -> grant
        assert r.available == 0
        assert r.busy == 5
        assert r.waiting == 0

    def test_request_quantity_negative_defaults_to_one(self):
        from dynafx.system.des import Resource

        r = Resource("r", capacity=5)
        assert r.request(0.0, quantity=-1) is True
        assert r.busy == 1

    def test_stats_count_requests(self):
        from dynafx.system.des import Resource

        r = Resource("r", capacity=2)
        r.request(0.0)
        r.request(0.0, quantity=2)
        r.request(0.0)
        assert r.stats.total_requests == 3
        assert r.stats.total_granted == 2
        assert r.stats.total_denied == 1

    def test_release_no_auto_grant_when_quantity_too_large(self):
        from dynafx.system.des import Resource

        r = Resource("r", capacity=10)
        r.request(0.0, quantity=10)
        r.request(0.0, wait=True, quantity=8)
        assert r.waiting == 1
        r.release(1.0)
        # Only 1 available, waiter needs 8
        assert r.waiting == 1
        for _ in range(7):
            r.release(2.0)
        # After 8 total releases: busy=10-8=2, auto-grant adds waiter's 8 -> busy=10
        assert r.waiting == 0
        assert r.busy == 10

    def test_request_catches_hook_exception(self):
        from dynafx.registry import register_resource_hook
        from dynafx.system.des import Resource

        def bad_hook(**kw):
            raise RuntimeError("not available")

        register_resource_hook("pre_request", bad_hook)
        r = Resource("r", capacity=5)
        assert r.request(0.0) is False
        assert r.busy == 0
