#!/usr/bin/env python3
"""
SaaS Churn Signal Chain — Google Trends as Leading Indicator

Demonstrates: "Ugly searches" anticipate churn before revenue drops.
Shows: scenarios, sensitivity, causal tracing, feedback loop detection.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.environ.setdefault('TMPDIR', '/tmp')

from dynafx.system.dsl import parse_sysd_file
from dynafx.system.feedback import detect_feedback_loops
from dynafx.system.causal import causal_trace
from dynafx.system.units import UnitChecker


def main():
    model = parse_sysd_file(os.path.join(os.path.dirname(__file__), '..', 'models', 'saas_churn_signal.sysd'))
    print(f'Model: {model.name}')
    print(f'Stocks: {[s.name for s in model.stocks]}')
    print()

    # ── Baseline ──────────────────────────────────────────────────
    base_params = {
        'base_negativity': 0.15, 'crisis_intensity': 0.8, 'crisis_start': 60, 'crisis_duration': 14,
        'signal_detection_delay': 30, 'sentiment_sensitivity': 1.5, 'sentiment_delay': 21,
        'base_churn': 0.003, 'churn_threshold': 0.003, 'pricing_sensitivity': 3.0, 'pricing_delay': 7,
        'base_acquisition': 15, 'max_acquisition': 40, 'wom_multiplier': 2.0,
        'base_ARPU': 50, 'arpu_discount_factor': 0.6,
        'investment_lag': 30, 'quality_sensitivity': 5.0,
        'cost_per_churned_user': 200, 'investment_pct': 0.25,
        'churn_lag': 21,
    }

    r = model.simulate(params=base_params)
    print('=== Baseline Run ===')
    print(f'  Users: {r.values["Active_Users"][0]:.0f} -> {r.values["Active_Users"][-1]:.0f}')
    print(f'  Cumulative churn: {r.values["Cumulative_Churn"][-1]:.0f}')

    # Compute all signal timestamps once (Fix 8: no duplicate)
    neg = r.values['Signal_Negativity']
    qual = r.values['Signal_Quality']
    sent = r.values['Signal_Sentiment']
    det = r.values['Signal_Detected']
    churn = [r.values['Active_Users'][i-1] - r.values['Active_Users'][i] if i > 0 else 0
             for i in range(len(r.values['Active_Users']))]

    peak_neg_t = r.times[neg.index(max(neg))]
    peak_det_t = r.times[det.index(max(det))]
    min_sent_t = r.times[sent.index(min(sent))]
    min_qual_t = r.times[qual.index(min(qual))]
    # Skip first 10 days to avoid initialization artifacts
    peak_churn_t = r.times[10 + churn[10:].index(max(churn[10:]))]
    print(f'  Signal: negativity peak day {peak_neg_t:.0f} -> detected day {peak_det_t:.0f} -> sentiment min day {min_sent_t:.0f} -> quality min day {min_qual_t:.0f}')
    print()

    # ── Scenarios ──────────────────────────────────────────────────
    print('=== Scenario Comparison ===')

    scenarios = {
        'Baseline': base_params,
        'Delayed Detection': {**base_params, 'signal_detection_delay': 45},
        'Strong Crisis': {**base_params, 'crisis_intensity': 1.5, 'crisis_duration': 28},
        'Fast Response': {**base_params, 'signal_detection_delay': 14, 'investment_lag': 15},
        'No Crisis': {**base_params, 'crisis_intensity': 0},
    }

    for name, params in scenarios.items():
        res = model.simulate(params=params)
        users_end = res.values['Active_Users'][-1]
        churn_total = res.values['Cumulative_Churn'][-1]
        min_q = min(res.values['Signal_Quality'])
        print(f'  {name:25s}  users={users_end:>8.0f}  churn={churn_total:>6.0f}  min_quality={min_q:.3f}')
    print()

    # ── Sensitivity Analysis ───────────────────────────────────────
    print('=== Sensitivity Analysis ===')
    print('  Varying each parameter ±50% from baseline, measuring final user count:')
    print()

    key_params = [
        ('crisis_intensity', 'Crisis Intensity'),
        ('crisis_duration', 'Crisis Duration'),
        ('signal_detection_delay', 'Signal Detection Delay'),
        ('cost_per_churned_user', 'Retention Cost/User'),
        ('investment_lag', 'Investment Lag'),
        ('quality_sensitivity', 'Quality Sensitivity'),
        ('pricing_sensitivity', 'Pricing Sensitivity'),
        ('base_churn', 'Base Churn Rate'),
    ]

    for param_name, label in key_params:
        low = {**base_params, param_name: base_params[param_name] * 0.5}
        high = {**base_params, param_name: base_params[param_name] * 1.5}
        r_low = model.simulate(params=low)
        r_high = model.simulate(params=high)
        users_low = r_low.values['Active_Users'][-1]
        users_high = r_high.values['Active_Users'][-1]
        delta = users_high - users_low
        direction = '+' if delta > 0 else ''
        if abs(delta) > 10:
            print(f'  {label:30s}  low={users_low:>7.0f}  high={users_high:>7.0f}  delta={direction}{delta:.0f}')
    print()

    # ── Feedback Loops ─────────────────────────────────────────────
    print('=== Feedback Loop Detection ===')
    analysis = detect_feedback_loops(model)
    if analysis.loops:
        # Fix 7: only multi-node loops are meaningful
        meaningful = [l for l in analysis.loops if len(l.nodes) > 1]
        for loop in meaningful:
            polarity = 'Reinforcing (R)' if loop.polarity == 'reinforcing' else 'Balancing (B)'
            print(f'  Loop: {loop.name}')
            print(f'    Type: {polarity}')
            print(f'    Nodes: {" -> ".join(loop.nodes[:6])}{"..." if len(loop.nodes) > 6 else ""}')
            print()
    else:
        print('  No feedback loops detected')
        print()

    # ── Causal Trace ───────────────────────────────────────────────
    print('=== Causal Trace: Active_Users ===')
    state_dict = {name: vals[-1] for name, vals in r.values.items()}
    trace = causal_trace(model, 'Active_Users', state_dict)
    print('  Causes (upstream dependencies):')
    causes = trace['causes']
    if causes:
        for child in causes.get('children', [])[:8]:
            print(f'    - {child["name"]}: {child["expr"][:60]}')
    print()
    print('  Effects (downstream consequences):')
    effects = trace['effects']
    if effects:
        for child in effects.get('children', [])[:8]:
            print(f'    - {child["name"]}: {child["expr"][:60]}')
    print()

    # ── Signal Lead Time Analysis ──────────────────────────────────
    print('=== Signal Lead Time Analysis ===')
    print(f'  Google Trends peak:     day {peak_neg_t:.0f} (intensity={max(neg):.3f})')
    print(f'  Detection peak:         day {peak_det_t:.0f} (intensity={max(det):.3f})')
    print(f'  Churn peak:             day {peak_churn_t:.0f} (max churn={max(churn):.1f} users/day)')
    print(f'  Sentiment minimum:      day {min_sent_t:.0f} (level={min(sent):.3f})')
    print(f'  Quality minimum:        day {min_qual_t:.0f} (level={min(qual):.3f})')
    print()
    # Fix 5: head-start is Google Trends peak -> churn spike (the useful warning window)
    head_start = peak_churn_t - peak_neg_t
    print(f'  Google Trends -> Churn:  {head_start:.0f} days (the warning window)')
    print(f'  Google Trends -> Detection: {peak_det_t - peak_neg_t:.0f} days')
    print(f'  Detection -> Churn:      {peak_churn_t - peak_det_t:.0f} days')
    print()

    # ── Unit Check ─────────────────────────────────────────────────
    print('=== Units Check ===')
    checker = UnitChecker()
    result = checker.check(model)
    if result.passed:
        print('  All units consistent.')
    else:
        for v in result.violations:
            print(f'  WARNING: {v}')
    print()

    # ── Key Insight ────────────────────────────────────────────────
    print('=== Key Insight ===')
    print('  The Google Trends negativity signal peaks BEFORE the churn')
    print('  impact hits revenue. Companies that monitor this signal')
    print(f'  have a {head_start:.0f}-day head start to respond.')
    print()
    print('  Response options:')
    print('    1. Proactive: address issues before churn spikes')
    print('    2. Retention offers: targeted discounts for at-risk users')
    print('    3. Product fixes: invest in quality before revenue drops')


if __name__ == '__main__':
    main()
