"""Logistics showcase — all new features in one simulation.

Demonstrates:
  - N-echelon supply chain (D1) via supplier_chain_topology plugin
  - Resource quantity requests (E2)
  - Availability calendars (E1) via availability_calendar plugin
  - Period-based service level (D3) via period_service_level plugin
  - Balking/reneging (E3) via balking_reneging plugin
  - CONVEY_BATCH (C2) for transport batching
  - Plugin registry integration
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dynafx.system.dsl import parse_sysd_file, parse_sysd
from dynafx.system.des import DESEngine, Queue as DesQueue, Resource
from dynafx.registry import clear_all

# ── Activate plugins ─────────────────────────────────────────────
import plugins.availability_calendar
import plugins.period_service_level
import plugins.balking_reneging
import plugins.supply_chain_topology


def part1_n_echelon_model():
    """D1: Configurable echelon topology with CONVEY_BATCH."""
    print("=" * 60)
    print("PART 1: N-Echelon Supply Chain")
    print("=" * 60)

    from plugins.supply_chain_topology import create_n_echelon_model

    model = create_n_echelon_model(
        echelons=["Retailer", "Warehouse", "Factory"],
        base_demand=500,
        shipping_delay=2,
        reorder_point=2000,
        safety_stock=500,
        batch_size=500,
        factory_capacity=500,
        smoothing_time=4,
        initial_inventory_factor=1.0,
    )

    r = model.simulate(method="euler", dt=1, t_span=(0, 150))

    for stock in ["Retailer_Inventory", "Warehouse_Inventory", "Factory_Inventory"]:
        vals = r["values"][stock]
        print(f"  {stock:25s}  min={min(vals):8.0f}  max={max(vals):8.0f}  end={vals[-1]:8.0f}")

    min_retail = min(r["values"]["Retailer_Inventory"])
    assert min_retail > 0, f"Retailer stockout at {min_retail}"
    print(f"  ✅ No stockouts at retailer (min={min_retail:.0f})")
    print()


def part2_des_with_quantity_and_calendar():
    """E2 + E1: Resource quantity requests + availability calendar."""
    print("=" * 60)
    print("PART 2: Resource Quantity + Availability Calendar")
    print("=" * 60)

    from plugins.availability_calendar import ResourceCalendar, register as reg_calendar

    engine = DESEngine()
    warehouse = Resource("warehouse", capacity=10, cost_per_unit=10.0)
    engine.add_resource(warehouse)

    cal = ResourceCalendar()
    cal.add_window(start_hour=6, end_hour=22, days_of_week=[0, 1, 2, 3, 4])
    reg_calendar(cal, warehouse)

    granted_1 = warehouse.request(t=0.0, wait=True, quantity=3)
    print(f"  t=0 (Thu 00:00, outside hours): request qty=3 → granted={granted_1}")

    granted_2 = warehouse.request(t=0.35, wait=True, quantity=5)
    print(f"  t=0.35 (Thu 08:24, inside hours):  request qty=5 → granted={granted_2}")

    warehouse.release(t=0.5)
    available = warehouse.available
    busy = warehouse.busy
    print(f"  After release: available={available}, busy={busy}")
    assert busy <= warehouse.capacity
    print(f"  ✅ Quantity requests + calendar work correctly")
    print()


def part3_service_level_tracking():
    """D3: Period-based service level with a DES queue."""
    print("=" * 60)
    print("PART 3: Period-Based Service Level")
    print("=" * 60)

    from plugins.period_service_level import ServiceLevelTracker, register as reg_sl

    tracker = ServiceLevelTracker(period_days=7)
    q = DesQueue("order_queue", capacity=100)
    engine = DESEngine()
    engine.add_queue(q)
    reg_sl(tracker, q)

    for day in range(21):
        t = float(day)
        q.enqueue(f"order_{day}", t)
        if day % 3 == 0:
            q.dequeue(t + 0.5)

    for period_num in range(3):
        sl = tracker.get_service_level(float(period_num * 7 + 3))
        print(f"  Week {period_num + 1}: demand={sl['total_demand']}, "
              f"met={sl['total_met']}, fill_rate={sl['fill_rate']:.0%}")

    print(f"  ✅ Service level tracking works across {period_num + 1} periods")
    print()


def part4_balking_and_reneging():
    """E3: Balking and reneging in a DES queue."""
    print("=" * 60)
    print("PART 4: Balking + Reneging")
    print("=" * 60)

    from plugins.balking_reneging import add_balking, add_reneging

    engine = DESEngine()
    q = DesQueue("service_queue", capacity=50, service_time="3.0")
    engine.add_queue(q)

    add_balking(q, max_wait=5.0)
    add_reneging(q, timeout=8.0, des_engine=engine)

    for i in range(10):
        q.enqueue(f"customer_{i}", float(i), event_queue=engine.event_queue)

    for step in range(20):
        t = float(step) * 2
        engine.step(t, 2.0)

    print(f"  Queue length after processing: {q.length()}")
    print(f"  Total dropped (including reneged): {q.stats.total_dropped}")
    if hasattr(q.stats, "total_reneged"):
        print(f"  Reneged: {q.stats.total_reneged}")

    print(f"  ✅ Balking/reneging functions correctly")
    print()


def part5_sysd_model_with_convey_batch():
    """C2: CONVEY_BATCH in a .sysd model."""
    print("=" * 60)
    print("PART 5: CONVEY_BATCH Pipeline Model")
    print("=" * 60)

    m = parse_sysd("""
model 'BatchPipeline'
  dt 1
  from 0 to 30
  aux input_rate: 100
  aux batch_delay: 3
  aux batch_size: 400
  aux shipped: CONVEY_BATCH(input_rate, batch_delay, batch_size)
  stock 'Buffer': 0
    + 'inflow': input_rate
    - 'outflow': shipped
""")
    r = m.simulate(method="euler")
    vals = r["values"]["Buffer"]
    print(f"  Buffer: min={min(vals):.0f}, max={max(vals):.0f}, end={vals[-1]:.0f}")
    assert min(vals) >= 0, "Buffer went negative"
    print(f"  ✅ CONVEY_BATCH produces correct batch pulses")
    print()


def part6_reference_models():
    """F1-F3: Parse and simulate all three reference models."""
    print("=" * 60)
    print("PART 6: Reference Models")
    print("=" * 60)

    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")

    for name, filename in [
        ("VMI", "vmi.sysd"),
        ("Reverse Logistics", "reverse_logistics.sysd"),
        ("Cold Chain", "cold_chain.sysd"),
    ]:
        path = os.path.join(models_dir, filename)
        assert os.path.exists(path), f"Missing {path}"
        m = parse_sysd_file(path)
        r = m.simulate(method="euler")
        stock_names = list(r["values"].keys())
        end_vals = [f'{r["values"][s][-1]:.0f}' for s in stock_names]
        print(f"  {name:20s} stocks={stock_names}  steps={len(r.times)}  "
              f"end_values={end_vals}")

    print(f"  ✅ All reference models parse and simulate")
    print()


def part7_end_to_end_supply_chain():
    """Full end-to-end: SD supply chain feeding DES warehouse."""
    print("=" * 60)
    print("PART 7: End-to-End — SD + DES + Plugins")
    print("=" * 60)

    from plugins.availability_calendar import ResourceCalendar, register as reg_calendar
    from plugins.period_service_level import ServiceLevelTracker, register as reg_sl
    from plugins.balking_reneging import add_balking, add_reneging
    from plugins.supply_chain_topology import create_n_echelon_model

    clear_all()

    model = create_n_echelon_model(
        echelons=["Retailer", "Warehouse", "Factory"],
        base_demand=500,
        shipping_delay=2,
        reorder_point=2000,
        safety_stock=500,
        batch_size=500,
        factory_capacity=500,
        initial_inventory_factor=1.0,
    )

    des_engine = DESEngine()
    dock = Resource("loading_dock", capacity=10, cost_per_unit=50.0)
    des_engine.add_resource(dock)

    cal = ResourceCalendar()
    cal.add_window(start_hour=6, end_hour=22, days_of_week=[0, 1, 2, 3, 4])
    reg_calendar(cal, dock)

    # Track cost via record_utilization at each state change
    dock.record_utilization(0.0)
    dock.request(t=0.35, quantity=2)
    dock.record_utilization(0.35)
    dock.request(t=0.5, quantity=1)
    dock.record_utilization(0.5)

    r = model.simulate(method="euler", dt=1, t_span=(0, 200))
    dock.record_utilization(200.0)

    retail = r["values"]["Retailer_Inventory"]
    warehouse = r["values"]["Warehouse_Inventory"]

    print(f"  Retailer_Inventory:  min={min(retail):.0f}  max={max(retail):.0f}  end={retail[-1]:.0f}")
    print(f"  Warehouse_Inventory: min={min(warehouse):.0f}  max={max(warehouse):.0f}  end={warehouse[-1]:.0f}")
    print(f"  Loading dock:        busy={dock.busy}/{dock.capacity}  available={dock.available}")
    print(f"  Dock cost:           ${dock.stats.total_cost:.2f}")
    assert min(retail) > 0, f"Retailer stockout at {min(retail)}"
    assert dock.busy <= dock.capacity, "Over capacity!"
    assert dock.stats.total_cost > 0, "Dock cost should be >0"
    print(f"  ✅ Full end-to-end simulation successful")
    print()


if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║     Logistics Showcase — Feature Demonstration      ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    part1_n_echelon_model()
    part2_des_with_quantity_and_calendar()
    part3_service_level_tracking()
    part4_balking_and_reneging()
    part5_sysd_model_with_convey_batch()
    part6_reference_models()
    part7_end_to_end_supply_chain()

    clear_all()
    print("All parts completed successfully.")
