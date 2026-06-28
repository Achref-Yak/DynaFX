"""N-echelon supply chain model builder.

Usage:
    from plugins.supply_chain_topology import create_n_echelon_model

    model = create_n_echelon_model(
        echelons=["Retailer", "Warehouse", "Factory"],
        base_demand=500,
        shipping_delay=6,
        reorder_point=2000,
        safety_stock=500,
    )
    result = model.simulate(params={"base_demand": 500})
"""

from typing import Any

from dynafx.system.dsl import AuxDef, FlowDef, StockDef, SysdModel


def create_n_echelon_model(
    echelons: list[str],
    base_demand: float = 500.0,
    shipping_delay: float = 6.0,
    reorder_point: float = 2000.0,
    safety_stock: float = 500.0,
    smoothing_time: float = 4.0,
    batch_size: float = 100.0,
    factory_capacity: float = 600.0,
    initial_inventory_factor: float = 2.0,
    **kwargs: Any,
) -> SysdModel:
    model = SysdModel(name=f"{'_'.join(echelons)}_SupplyChain")
    model.dt = 1.0
    model.t_span = (0.0, 200.0)

    model.aux_vars.append(AuxDef(name="base_demand", expr=str(base_demand)))
    model.aux_vars.append(AuxDef(name="shipping_delay", expr=str(shipping_delay)))
    model.aux_vars.append(AuxDef(name="reorder_point", expr=str(reorder_point)))
    model.aux_vars.append(AuxDef(name="safety_stock", expr=str(safety_stock)))
    model.aux_vars.append(AuxDef(name="smoothing_time", expr=str(smoothing_time)))
    model.aux_vars.append(AuxDef(name="batch_size", expr=str(batch_size)))
    model.aux_vars.append(AuxDef(name="factory_capacity", expr=str(factory_capacity)))

    n = len(echelons)

    for i, name in enumerate(echelons):
        inv_name = f"{name}_Inventory"
        init_val = (reorder_point + safety_stock) * initial_inventory_factor
        init_val = max(init_val, 100.0)

        raw_demand_name = f"{name}_raw_demand"
        if i == 0:
            model.aux_vars.append(AuxDef(name=raw_demand_name, expr="base_demand"))
        else:
            prev_name = echelons[i - 1]
            model.aux_vars.append(AuxDef(
                name=raw_demand_name,
                expr=f"{prev_name}_order_smoothed",
            ))

        order_name = f"{name}_order"
        model.aux_vars.append(AuxDef(
            name=order_name,
            expr=f"MAX(0, (reorder_point + safety_stock - {inv_name}) / smoothing_time + {raw_demand_name})",
        ))

        smoothed_name = f"{name}_order_smoothed"
        model.aux_vars.append(AuxDef(
            name=smoothed_name,
            expr=f"SMOOTHI({order_name}, smoothing_time, base_demand)",
        ))

        flows: list[FlowDef] = []

        if i == 0:
            out_name = f"{name}_out"
            flows.append(FlowDef(name=f"- {out_name}", direction="-", expr=f"MIN({inv_name}, base_demand)"))
        else:
            ship_name = f"{name}_ships"
            flows.append(FlowDef(
                name=f"- {ship_name}",
                direction="-",
                expr=f"MIN({inv_name}, {smoothed_name})",
            ))

        if i == n - 1:
            prod_name = f"{name}_production"
            flows.append(FlowDef(
                name=f"+ {prod_name}",
                direction="+",
                expr=f"MIN(factory_capacity, {inv_name} + 500)",
            ))
        else:
            shipped_name = f"from_{echelons[i + 1]}_to_{name}"
            upstream_stock = f"{echelons[i + 1]}_Inventory"
            upstream_order = f"{echelons[i + 1]}_order_smoothed"
            flows.append(FlowDef(
                name=f"+ {shipped_name}",
                direction="+",
                expr=f"CONVEY_BATCH(MIN({upstream_stock}, {upstream_order}), shipping_delay, batch_size)",
            ))

        model.stocks.append(StockDef(
            name=inv_name,
            initial=init_val,
            flows=flows,
        ))

    return model
