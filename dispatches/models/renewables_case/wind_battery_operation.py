from dispatches.models.renewables_case.wind_battery_double_loop import (
    create_multiperiod_wind_battery_model,
    transform_design_model_to_operation_model,
)
import pyomo.environ as pyo


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
