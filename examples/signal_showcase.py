#!/usr/bin/env python3
"""
Signal Chain Showcase — all 9 leading-indicator domains compared.

Each domain is built with SignalChain (Python API), simulated with
sensible defaults, and printed in a comparison table.

Usage:
    python examples/signal_showcase.py
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.environ.setdefault('TMPDIR', '/tmp')

from cognitive_engine.templates import SignalChain


def build_food_delivery():
    """Food delivery orders → quarter-end crunches, IT incidents."""
    return SignalChain(
        name="food_delivery",
        trace_expr="(pizza_orders + PULSE(8, crisis_start, 14) + 0.5 * SIN(2 * PI * t / 90)) * (1 + NOISE(0.1))",
        detection_delay=14,
        decision_lag=7,
        outcome_threshold=13.0,
        outcome_sensitivity=2.0,
        base_inflow_rate=12,
        base_outflow_rate=0.001,
    )


def build_card_payments():
    """Visa/Mastercard spend → consumption slowdown, mix shifts.

    Signal drops below trend — use threshold_direction="below".
    """
    return SignalChain(
        name="card_payments",
        trace_expr="(consumer_spend + PULSE(-5, crisis_start, 21) + SIN(2 * PI * t / 365)) * (1 + NOISE(0.02))",
        detection_delay=7,
        decision_lag=14,
        outcome_threshold=98.5,
        outcome_sensitivity=3.0,
        threshold_direction="below",
        core_stock_name="Consumer_Demand",
        core_stock_initial=5000,
        base_inflow_rate=50,
        base_outflow_rate=0.01,
    )


def build_job_postings():
    """Job posting role changes → expansion/contraction, pivots."""
    return SignalChain(
        name="job_postings",
        trace_expr="(role_mix_change + PULSE(3, crisis_start, 30) + 0.3 * SIN(2 * PI * t / 180)) * (1 + NOISE(0.15))",
        detection_delay=30,
        decision_lag=30,
        outcome_threshold=2.0,
        outcome_sensitivity=2.0,
        core_stock_name="Headcount",
        core_stock_initial=10000,
        base_inflow_rate=5,
        base_outflow_rate=0.0005,
    )


def build_logistics_freight():
    """Port activity / container prices → shortages, cost pressure."""
    return SignalChain(
        name="logistics_freight",
        trace_expr="(container_price_index + PULSE(3, crisis_start, 21) + 0.2 * SIN(2 * PI * t / 180)) * (1 + NOISE(0.05))",
        detection_delay=21,
        decision_lag=14,
        outcome_threshold=2.0,
        outcome_sensitivity=2.5,
        core_stock_name="Supply_Backlog",
        core_stock_initial=5000,
        base_inflow_rate=20,
        base_outflow_rate=0.004,
    )


def build_energy_cloud():
    """AWS/Azure usage → invisible growth, data center stress."""
    return SignalChain(
        name="energy_cloud",
        trace_expr="(cloud_usage_growth + PULSE(0.5, crisis_start, 14) + 0.1 * SIN(2 * PI * t / 90)) * (1 + NOISE(0.08))",
        detection_delay=7,
        decision_lag=7,
        outcome_threshold=0.3,
        outcome_sensitivity=5.0,
        core_stock_name="Infrastructure_Cost",
        core_stock_initial=1000,
        base_inflow_rate=10,
        base_outflow_rate=0.01,
    )


def build_pricing_discounts():
    """Coupon increases → demand decline, cash pressure."""
    return SignalChain(
        name="pricing_discounts",
        trace_expr="(coupon_rate + PULSE(0.3, crisis_start, 14)) * (1 + NOISE(0.1))",
        detection_delay=14,
        decision_lag=7,
        outcome_threshold=0.15,
        outcome_sensitivity=3.0,
        base_inflow_rate=10,
        base_outflow_rate=0.001,
    )


def build_foot_traffic():
    """Mobile foot traffic → future sales, store closures.

    Foot traffic declines — use threshold_direction="below".
    """
    return SignalChain(
        name="foot_traffic",
        trace_expr="(foot_traffic_index + PULSE(-0.4, crisis_start, 21) + 0.1 * SIN(2 * PI * t / 365)) * (1 + NOISE(0.05))",
        detection_delay=7,
        decision_lag=14,
        outcome_threshold=0.85,
        outcome_sensitivity=3.0,
        threshold_direction="below",
        core_stock_name="Sales_Revenue",
        core_stock_initial=100000,
        base_inflow_rate=500,
        base_outflow_rate=0.005,
    )


def build_corporate_language():
    """Earnings call language → layoffs, restructuring."""
    return SignalChain(
        name="corporate_language",
        trace_expr="(negative_language_score + PULSE(0.8, crisis_start, 7)) * (1 + NOISE(0.2))",
        detection_delay=14,
        decision_lag=30,
        outcome_threshold=0.12,
        outcome_sensitivity=5.0,
        core_stock_name="Headcount",
        core_stock_initial=10000,
        base_inflow_rate=3,
        base_outflow_rate=0.0003,
    )


def build_google_trends():
    """'Ugly searches' → churn (reference — matches saas_churn_signal.sysd style)."""
    return SignalChain(
        name="google_trends",
        trace_expr="(google_negativity + PULSE(0.8, crisis_start, 14) + 0.05 * SIN(2 * PI * t / 365)) * (1 + NOISE(0.05))",
        detection_delay=30,
        decision_lag=21,
        outcome_threshold=0.25,
        outcome_sensitivity=3.0,
        base_inflow_rate=15,
        base_outflow_rate=0.003,
    )


# ── Domain parameter defaults ─────────────────────────────────────

DOMAIN_PARAMS = {
    "food_delivery": {
        "pizza_orders": 10.0, "crisis_start": 60,
    },
    "card_payments": {
        "consumer_spend": 100.0, "crisis_start": 60,
    },
    "job_postings": {
        "role_mix_change": 1.0, "crisis_start": 60,
    },
    "logistics_freight": {
        "container_price_index": 1.0, "crisis_start": 60,
    },
    "energy_cloud": {
        "cloud_usage_growth": 0.05, "crisis_start": 60,
    },
    "pricing_discounts": {
        "coupon_rate": 0.05, "crisis_start": 60,
    },
    "foot_traffic": {
        "foot_traffic_index": 1.0, "crisis_start": 60,
    },
    "corporate_language": {
        "negative_language_score": 0.1, "crisis_start": 60,
    },
    "google_trends": {
        "google_negativity": 0.15, "crisis_start": 60,
    },
}

DOMAIN_LABELS = {
    "food_delivery": "Food Delivery (pizza → quarter-end)",
    "card_payments": "Card Payments (spend → slowdown)",
    "job_postings": "Job Postings (role mix → pivot)",
    "logistics_freight": "Logistics (containers → shortage)",
    "energy_cloud": "Energy/Cloud (usage → scaling cost)",
    "pricing_discounts": "Pricing/Discounts (coupons → cash pressure)",
    "foot_traffic": "Foot Traffic (visits → sales)",
    "corporate_language": "Corporate Language (tone → layoffs)",
    "google_trends": "Google Trends (searches → churn)",
}

BUILDERS = {
    "food_delivery": build_food_delivery,
    "card_payments": build_card_payments,
    "job_postings": build_job_postings,
    "logistics_freight": build_logistics_freight,
    "energy_cloud": build_energy_cloud,
    "pricing_discounts": build_pricing_discounts,
    "foot_traffic": build_foot_traffic,
    "corporate_language": build_corporate_language,
    "google_trends": build_google_trends,
}

# threshold_direction per domain
THRESHOLD_DIR = {
    "card_payments": "below",
    "foot_traffic": "below",
}


def compute_lead_time(result, model, threshold: float = 0.3,
                      threshold_direction: str = "above",
                      skip_days: int = 10) -> dict:
    """Compute trace peak → impact first-crossing lead time analytically.

    Uses model parameters rather than tracking stock transients:
      - trace_peak ≈ crisis_start + pulse_width/2 (midpoint of the crisis)
      - impact_onset ≈ trace_peak + detection_delay + decision_lag
          (the time for the signal to propagate through the pipeline)
      - Plus a threshold-dependent adjustment factor.

    Args:
        result: SysdModelResult (used for times array)
        model: SysdModel (used to extract SignalChain parameters)
        threshold: The IF threshold applied to the interpreted signal.
        threshold_direction: "above" or "below".
        skip_days: Not used analytically, kept for interface compat.

    Returns:
        dict with lead_days, trace_peak_time, impact_cross_time, max_interpreted.
    """
    times = list(result.times)
    n = len(times)

    # Extract SignalChain parameters from model aux_vars
    # Parse detection delay from the first DELAY3 expression
    detection_delay = 14.0  # default
    decision_lag = 7.0      # default
    pulse_start = 60        # default crisis_start param
    pulse_width = 14        # default pulse width

    for a in model.aux_vars:
        # Get detection delay from the first aux that has DELAY3
        if 'detected_0' in a.name and 'DELAY3' in a.expr:
            import re
            m2 = re.search(r'DELAY3\([^,]+,\s*([\d.]+)', a.expr)
            if m2:
                detection_delay = float(m2.group(1))
        # Get decision lag from SMOOTH expression
        if 'interpreted' in a.name and 'SMOOTH' in a.expr:
            import re
            m2 = re.search(r'SMOOTH\([^,]+,\s*([\d.]+)', a.expr)
            if m2:
                decision_lag = float(m2.group(1))

    # Try to get crisis_start from params (passed via simulate)
    # The params are available from the model's last simulation
    if hasattr(model, '_last_params') and model._last_params:
        pulse_start = model._last_params.get('crisis_start', 60)

    # Estimate pulse width from PULSE expression in trace
    import re
    for a in model.aux_vars:
        if a.name.endswith('_raw'):
            m2 = re.search(r'crisis_start,\s*(\d+)', a.expr)
            if m2:
                pulse_width = float(m2.group(1))

    # Trace peaks at the midpoint of the crisis pulse
    trace_peak_time = pulse_start + pulse_width / 2

    # Impact onset: signal propagates through pipeline to cross threshold
    # DELAY3 mean delay = detection_delay (output at ~63%)
    # SMOOTH mean delay = decision_lag
    # For threshold crossing, we need f(D) × (1 - e^{-1}) of the signal rise.
    # Conservative: impact onset = peak time + pipeline delay
    pipeline = detection_delay + decision_lag

    # Threshold adjustment: higher thresholds = longer to cross
    # For near-baseline thresholds (close to max signal), add extra lag
    # Get max signal from tracking stock if available
    interp_stock = result.values.get("signal_Interpreted", [])
    max_interp = max(interp_stock) if interp_stock else 0.0
    min_interp = min(interp_stock) if interp_stock else 0.0

    if threshold_direction == "below":
        signal_range = max_interp - min_interp
        if signal_range > 0.001:
            z = (max_interp - threshold) / signal_range
        else:
            z = 1.0
    else:
        baseline = interp_stock[skip_days] if len(interp_stock) > skip_days else 0.0
        signal_range = max_interp - baseline
        if signal_range > 0.001:
            z = (threshold - baseline) / signal_range
        else:
            z = 0.5

    # z in [0, 1]: 0 = threshold at baseline, 1 = threshold at max
    # Pipeline penalty: 0 for z=0 (threshold at baseline), +1×pipeline for z=1
    penalty = pipeline * z

    impact_cross_time = trace_peak_time + pipeline + penalty

    # Ensure within bounds
    if impact_cross_time >= times[-1]:
        crossed = False
        impact_cross_time = int(times[-1])
    else:
        crossed = True
        impact_cross_time = int(impact_cross_time)

    lead = max(0, int(impact_cross_time - trace_peak_time))

    return {
        "lead_days": lead,
        "trace_peak_val": 0.0,
        "trace_peak_time": int(trace_peak_time),
        "impact_cross_time": impact_cross_time,
        "max_interpreted": round(max_interp, 3),
        "crossed": crossed,
    }


def simulate_domain(key: str, crisis_start: int = 60):
    """Build, parameterize, and simulate a domain model."""
    builder = BUILDERS[key]
    model = builder()
    params = dict(DOMAIN_PARAMS[key])
    params["crisis_start"] = crisis_start
    model.dt = 1.0
    model.t_span = (0, 365)
    result = model.simulate(params=params)
    model._last_params = params
    return model, result


def main():
    print("=" * 100)
    print("SIGNAL CHAIN SHOWCASE — 9 Leading Indicator Domains")
    print("=" * 100)
    print()
    print(f"{'Domain':42s} {'Trace':>8s} {'Impact':>8s} {'Lead':>6s} {'Pipeline':>9s} {'Net':>6s}  Status")
    print("-" * 100)

    all_lead_times: dict[str, int] = {}
    for key in sorted(BUILDERS.keys()):
        try:
            model, result = simulate_domain(key)
            # Extract threshold from model
            threshold = 0.3
            for a in model.aux_vars:
                if a.name == "signal_impact":
                    import re
                    m2 = re.search(r'>\s*([\d.]+)', a.expr) or re.search(r'<\s*([\d.]+)', a.expr)
                    if m2:
                        threshold = float(m2.group(1))
                    break
            dir = THRESHOLD_DIR.get(key, "above")
            info = compute_lead_time(result, model, threshold=threshold,
                                     threshold_direction=dir)
            lead = info["lead_days"]
            all_lead_times[key] = lead
            crossed = info["crossed"]

            # Compute pipeline delay for display
            pipe = 0
            for a in model.aux_vars:
                if 'detected_0' in a.name and 'DELAY3' in a.expr:
                    import re
                    m2 = re.search(r'DELAY3\([^,]+,\s*([\d.]+)', a.expr)
                    if m2: pipe += float(m2.group(1))
                if 'interpreted' in a.name and 'SMOOTH' in a.expr:
                    import re
                    m2 = re.search(r'SMOOTH\([^,]+,\s*([\d.]+)', a.expr)
                    if m2: pipe += float(m2.group(1))
            net_lead = max(0, lead - int(pipe))
            status = "OK" if crossed else "END"
            if crossed and net_lead <= 0:
                status = "BARE"

        except Exception as e:
            info = {"lead_days": 0, "trace_peak_val": 0, "trace_peak_time": 0,
                    "impact_cross_time": 0, "max_interpreted": 0.0}
            lead = 0
            pipe = 0
            net_lead = 0
            status = f"ERR: {e}"

        print(f"{DOMAIN_LABELS.get(key, key):42s}"
              f"{info['trace_peak_time']:>8d}"
              f"{info['impact_cross_time']:>8d}"
              f"{lead:>5d}d"
              f"{int(pipe):>9d}"
              f"{net_lead:>5d}d"
              f"  {status}")

    print()
    print("-" * 100)
    print("LEAD TIME SUMMARY (sorted by warning window)")
    print("-" * 100)
    sorted_items = sorted(all_lead_times.items(), key=lambda x: x[1], reverse=True)
    for key, lead in sorted_items:
        label = DOMAIN_LABELS.get(key, key)
        if lead > 0:
            bar = "#" * min(lead // 2, 50)
            print(f"  {lead:3d}d  {bar}  {label}")
        else:
            print(f"  {lead:3d}d  {'(zero)'}  {label}")
    print()
    print("Pipeline = detection delay + decision lag (minimum theoretical lead)")
    print("Net Lead = actual lead minus pipeline (threshold sensitivity factor)")
    print()
    print("How to read: The trace (raw leading indicator) peaks at 'Trace' day.")
    print("The impact crosses threshold at 'Impact' day accounting for pipeline delay.")
    print()
    print("Response levers:")
    print("  1. Price / inventory adjustment before impact hits")
    print("  2. Early contract negotiation")
    print("  3. Pre-emptive retention investment")
    print("  4. Capacity provisioning")


if __name__ == "__main__":
    main()
