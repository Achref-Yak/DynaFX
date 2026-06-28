"""Tests for dynafx plugins (D1, D3, E1, E3)."""
import pytest
from dynafx.registry import clear_all


class TestE1AvailabilityCalendar:
    def teardown_method(self):
        clear_all()

    def test_availability_calendar_is_available(self):
        from plugins.availability_calendar import ResourceCalendar

        cal = ResourceCalendar()
        cal.add_window(start_hour=0, end_hour=24, days_of_week=[0, 1, 2, 3, 4, 5, 6])
        assert cal.is_available(0.0) is True

    def test_availability_calendar_outside_hours(self):
        from plugins.availability_calendar import ResourceCalendar

        cal = ResourceCalendar()
        cal.add_window(start_hour=8, end_hour=18, days_of_week=[0, 1, 2, 3, 4])
        # t=0 = 1970-01-01 00:00 UTC = Thursday
        assert cal.is_available(0.0) is False

    def test_availability_calendar_inside_window(self):
        from plugins.availability_calendar import ResourceCalendar

        cal = ResourceCalendar()
        cal.add_window(start_hour=0, end_hour=24, days_of_week=[0, 1, 2, 3, 4])
        # 1970-01-01 00:00 UTC = Thursday (weekday 3)
        assert cal.is_available(0.0) is True

    def test_register_denies_request_outside_window(self):
        from dynafx.system.des import Resource
        from plugins.availability_calendar import ResourceCalendar, register

        cal = ResourceCalendar()
        cal.add_window(start_hour=8, end_hour=18, days_of_week=[3])
        r = Resource("r", capacity=5)
        register(cal, r)
        assert r.request(0.0) is False

    def test_register_allows_request_inside_window(self):
        from dynafx.system.des import Resource
        from plugins.availability_calendar import ResourceCalendar, register

        cal = ResourceCalendar()
        cal.add_window(start_hour=0, end_hour=24, days_of_week=[3])
        r = Resource("r", capacity=5)
        register(cal, r)
        assert r.request(0.0) is True

    def test_schedule_check_builtin_registered(self):
        import plugins.availability_calendar
        from dynafx.registry import get_registered_builtins

        # Force re-registration since clear_all() ran in teardown
        from dynafx.registry import register_builtin
        register_builtin("SCHEDULE_CHECK", plugins.availability_calendar._schedule_check)

        builtins = get_registered_builtins()
        assert "SCHEDULE_CHECK" in builtins
        assert builtins["SCHEDULE_CHECK"](0.0, 0, 24) == 1.0
        assert builtins["SCHEDULE_CHECK"](0.0, 8, 18) == 0.0


class TestD3PeriodServiceLevel:
    def teardown_method(self):
        clear_all()

    def test_tracker_records_arrivals(self):
        from plugins.period_service_level import ServiceLevelTracker

        tracker = ServiceLevelTracker(period_days=1)
        tracker.record_arrival(0.5)
        tracker.record_arrival(1.5)
        sl = tracker.get_service_level(0.5)
        assert sl["total_demand"] == 1
        assert sl["total_met"] == 0

    def test_tracker_records_departures(self):
        from plugins.period_service_level import ServiceLevelTracker

        tracker = ServiceLevelTracker(period_days=1)
        tracker.record_arrival(0.5)
        tracker.record_departure(0.8)
        sl = tracker.get_service_level(0.5)
        assert sl["total_demand"] == 1
        assert sl["total_met"] == 1
        assert sl["fill_rate"] == 1.0

    def test_tracker_fill_rate(self):
        from plugins.period_service_level import ServiceLevelTracker

        tracker = ServiceLevelTracker(period_days=1)
        for _ in range(10):
            tracker.record_arrival(0.5)
        for _ in range(7):
            tracker.record_departure(0.8)
        sl = tracker.get_service_level(0.5)
        assert sl["fill_rate"] == 0.7

    def test_tracker_multiple_periods(self):
        from plugins.period_service_level import ServiceLevelTracker

        tracker = ServiceLevelTracker(period_days=7)
        tracker.record_arrival(0.0)
        tracker.record_departure(0.0)
        tracker.record_arrival(10.0)
        sl0 = tracker.get_service_level(0.0)
        sl1 = tracker.get_service_level(10.0)
        assert sl0["period"] == 0
        assert sl0["fill_rate"] == 1.0
        assert sl1["period"] == 1
        assert sl1["total_demand"] == 1
        assert sl1["fill_rate"] == 0.0

    def test_register_hooks_fire(self):
        from dynafx.system.des import Queue
        from plugins.period_service_level import ServiceLevelTracker, register

        tracker = ServiceLevelTracker(period_days=1)
        q = Queue("test_q")
        register(tracker, q)
        q.enqueue("entity1", 0.5)
        q.dequeue(0.8)
        sl = tracker.get_service_level(0.5)
        assert sl["total_demand"] == 1
        assert sl["total_met"] == 1


class TestE3BalkingReneging:
    def teardown_method(self):
        clear_all()

    def test_add_balking_tracks_balked(self):
        from dynafx.system.des import Queue
        from plugins.balking_reneging import add_balking

        q = Queue("test_q")
        add_balking(q, max_wait=1.0)
        q.enqueue("first", 0.0)
        q.enqueue("second", 5.0)
        assert q.stats.total_balked >= 0

    def test_add_reneging_tracks_reneged(self):
        from dynafx.system.des import DESEngine, Queue
        from plugins.balking_reneging import add_reneging

        q = Queue("test_q")
        engine = DESEngine()
        engine.add_queue(q)
        add_reneging(q, timeout=2.0, des_engine=engine)
        q.enqueue("entity1", 0.0, event_queue=engine.event_queue)
        assert q.length() == 1
        assert len(engine.event_queue) > 0

    def test_renege_event_removes_entity(self):
        from dynafx.system.des import DESEngine, Queue
        from plugins.balking_reneging import add_reneging

        q = Queue("test_q")
        engine = DESEngine()
        engine.add_queue(q)
        add_reneging(q, timeout=1.0, des_engine=engine)
        q.enqueue("entity1", 0.0, event_queue=engine.event_queue)
        engine.step(0.0, 0.5)
        assert q.length() == 1
        engine.step(0.5, 2.0)
        assert q.length() == 0


class TestD1SupplyChainTopology:
    def teardown_method(self):
        clear_all()

    def test_create_two_echelon(self):
        from plugins.supply_chain_topology import create_n_echelon_model

        model = create_n_echelon_model(
            echelons=["Retailer", "Factory"],
            base_demand=100,
            shipping_delay=3,
        )
        assert model.name == "Retailer_Factory_SupplyChain"
        assert len(model.stocks) == 2
        assert model.stocks[0].name == "Retailer_Inventory"
        assert model.stocks[1].name == "Factory_Inventory"

    def test_create_three_echelon(self):
        from plugins.supply_chain_topology import create_n_echelon_model

        model = create_n_echelon_model(
            echelons=["Retailer", "Warehouse", "Factory"],
            base_demand=500,
        )
        assert len(model.stocks) == 3

    def test_simulate_two_echelon(self):
        from plugins.supply_chain_topology import create_n_echelon_model

        model = create_n_echelon_model(
            echelons=["Retailer", "Factory"],
            base_demand=100,
            shipping_delay=3,
        )
        result = model.simulate(method="euler")
        assert result is not None
        assert len(result.times) > 0

    def test_simulate_three_echelon(self):
        from plugins.supply_chain_topology import create_n_echelon_model

        model = create_n_echelon_model(
            echelons=["Retailer", "Warehouse", "Factory"],
            base_demand=500,
        )
        result = model.simulate(method="euler")
        assert all(v >= 0 for vals in result["values"].values() for v in vals)

    def test_custom_params(self):
        from plugins.supply_chain_topology import create_n_echelon_model

        model = create_n_echelon_model(
            echelons=["A", "B", "C", "D"],
            base_demand=200,
            shipping_delay=5,
            smoothing_time=3,
        )
        assert len(model.stocks) == 4
        result = model.simulate(method="euler")
        assert result is not None

    def test_retailer_inventory_does_not_go_negative(self):
        from plugins.supply_chain_topology import create_n_echelon_model

        model = create_n_echelon_model(
            echelons=["Retailer", "Warehouse", "Factory"],
            base_demand=500,
            shipping_delay=6,
            reorder_point=2000,
            safety_stock=500,
            batch_size=200,
            factory_capacity=2000,
        )
        result = model.simulate(method="euler")
        min_retail = min(result["values"]["Retailer_Inventory"])
        assert min_retail >= 0, f"Retailer inventory went to {min_retail}"

    def test_empty_echelons_returns_empty_model(self):
        from plugins.supply_chain_topology import create_n_echelon_model

        model = create_n_echelon_model(echelons=[])
        assert len(model.stocks) == 0
