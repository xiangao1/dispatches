from dispatches.models.renewables_case.wind_battery_double_loop import (
    create_multiperiod_wind_battery_model,
    transform_design_model_to_operation_model,
)
import pyomo.environ as pyo
from pyomo.core.expr.current import identify_variables


horizon = 24
model = create_multiperiod_wind_battery_model(n_time_points=horizon)
transform_design_model_to_operation_model(model)

# get the pyomo model
pyomo_model = model._pyomo_model

pyomo_model.Horizon = pyo.Set(initialize=range(horizon))

# power output in MW
pyomo_model.TotalPowerOutput = pyo.Expression(pyomo_model.Horizon)

# Operation costs in $
pyomo_model.OperationCost = pyo.Expression(pyomo_model.Horizon)

active_blks = model.get_active_process_blocks()
for (t, b) in enumerate(active_blks):
    pyomo_model.TotalPowerOutput[t] = (
        b.fs.splitter.grid_elec[0] + b.fs.battery.elec_out[0]
    ) * 1e-3
    pyomo_model.OperationCost[t] = b.fs.windpower.op_total_cost

# you can change the prices
pyomo_model.LMP = pyo.Param(pyomo_model.Horizon, initialize=20, mutable=True)

# deactivate objectives
for obj in pyomo_model.component_objects(pyo.Objective):
    obj.deactivate()

def max_profit_obj_rule(m):
    return sum(m.LMP[t] * m.TotalPowerOutput[t] - m.OperationCost[t] for t in m.Horizon)
pyomo_model.MaxProfitObj = pyo.Objective(rule=max_profit_obj_rule, sense=pyo.maximize)

solver = pyo.SolverFactory("cbc")
solver.solve(pyomo_model, tee=True)

# degrees of freedom: variables appeared in the TotalPowerOutput
# On each active block (element in `active_blks`)
for blk in active_blks:
    blk.fs.splitter.grid_elec[0]
    blk.fs.battery.elec_out[0]

# The blk.fs.splitter.grid_elec[0] appears in the following equality contraints
for blk in active_blks:
    blk.fs.splitter.sum_split

# The blk.fs.battery.elec_out[0] appears in the following equality contraints
for blk in active_blks:
    blk.fs.battery.accumulate_energy_throughput
    blk.fs.battery.state_evolution
