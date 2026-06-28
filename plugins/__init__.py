"""Plugin modules for dynafx.

Plugins extend the core framework with custom builtin functions and
DES hooks via the registry API. Import a plugin to activate it:

    import plugins.availability_calendar
    import plugins.balking_reneging
    import plugins.period_service_level
    import plugins.supply_chain_topology

Available plugins:
    availability_calendar  — shift-based resource availability
    balking_reneging       — entities leave queue after timeout
    period_service_level   — fill rate per time window (daily/weekly)
    supply_chain_topology  — N-echelon supply chain model builder
"""
