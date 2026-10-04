# Modelo MILP en Gurobi que asigna a cada dron monitorear, cargar o esperar.

import gurobipy as gp
from gurobipy import GRB


def resolver_ecoa_alert(data, verbose=False):
    D, G, H, T, Tm = data["D"], data["G"], data["H"], data["T"], data["Tm"]

    model = gp.Model("EcoAlert_Segunda_Implementacion")

    if not verbose:
        model.Params.OutputFlag = 0

    model.Params.MIPGap = 0.01
    model.Params.TimeLimit = 60

    # Variables
    z = model.addVars(D, G, Tm, vtype=GRB.BINARY, name="z")       # monitoreo
    q = model.addVars(D, Tm, vtype=GRB.BINARY, name="q")          # carga
    y = model.addVars(G, Tm, vtype=GRB.BINARY, name="y")          # sector monitoreado
    b = model.addVars(D, T, lb=0, vtype=GRB.CONTINUOUS, name="b")
    age = model.addVars(G, T, lb=0, vtype=GRB.CONTINUOUS, name="Age")
    s = model.addVars(G, T, lb=0, vtype=GRB.CONTINUOUS, name="s")
    batt_def = model.addVars(D, T, lb=0, vtype=GRB.CONTINUOUS, name="batt_def")

    miss_urgent = model.addVars(G, Tm, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="miss_urgent")
    miss_deuda = model.addVars(G, Tm, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="miss_deuda")

    # --------------------------------------------------------
    # Función objetivo
    # --------------------------------------------------------

    obj_age = gp.quicksum(
        data["w"][(g, min(t, max(Tm)))] * age[g, t]
        for g in G
        for t in T
    )

    obj_slack = gp.quicksum(s[g, t] for g in G for t in T)

    obj_travel = gp.quicksum(
        data["sortie_cost"][(d, g, t)] * z[d, g, t]
        for d in D
        for g in G
        for t in Tm
    )

    obj_batt = gp.quicksum(batt_def[d, t] for d in D for t in T)

    reward_monitor = gp.quicksum(
        data["w"][(g, t)] * z[d, g, t]
        for d in D
        for g in G
        for t in Tm
    )

    reward_critical = gp.quicksum(
        data["critical"][(g, t)] * z[d, g, t]
        for d in D
        for g in G
        for t in Tm
    )

    reward_deuda = gp.quicksum(
        data["deuda_operacional"][(g, t)] * z[d, g, t]
        for d in D
        for g in G
        for t in Tm
    )

    obj_urgent_miss = gp.quicksum(
        data["w"][(g, t)] * miss_urgent[g, t]
        for g in G
        for t in Tm
    )

    obj_deuda_miss = gp.quicksum(
        data["w"][(g, t)] * miss_deuda[g, t]
        for g in G
        for t in Tm
    )

    model.setObjective(
        obj_age
        + data["mu"] * obj_slack
        + data["lam"] * obj_travel
        + data["eta_batt"] * obj_batt
        + data["gamma_urgent_miss"] * obj_urgent_miss
        + data["gamma_deuda_miss"] * obj_deuda_miss
        - data["beta_monitor"] * reward_monitor
        - data["beta_critical"] * reward_critical
        - data["beta_deuda"] * reward_deuda,
        GRB.MINIMIZE
    )

    # --------------------------------------------------------
    # Condiciones iniciales
    # --------------------------------------------------------

    for d in D:
        model.addConstr(b[d, 0] == data["b0"][d], name=f"InitBatt_{d}")

    for g in G:
        model.addConstr(age[g, 0] == data["age0"][g], name=f"InitAge_{g}")

    # --------------------------------------------------------
    # Acciones por dron
    # --------------------------------------------------------

    for d in D:
        for t in Tm:
            model.addConstr(
                gp.quicksum(z[d, g, t] for g in G) + q[d, t] <= 1,
                name=f"OneAction_{d}_{t}"
            )

    # Carga o descanso forzado solo en condiciones operacionales específicas.
    for d in D:
        if data["force_charge0"].get(d, 0) == 1:
            model.addConstr(q[d, 0] == 1, name=f"ForceCharge0_{d}")

        if data["force_rest0"].get(d, 0) == 1:
            model.addConstr(
                gp.quicksum(z[d, g, 0] for g in G) == 0,
                name=f"ForceRest0_{d}"
            )

    # --------------------------------------------------------
    # Capacidad de carga por estación
    # --------------------------------------------------------

    for h in H:
        drones_en_h = [d for d in D if data["hub_base"][d] == h]

        for t in Tm:
            model.addConstr(
                gp.quicksum(q[d, t] for d in drones_en_h) <= data["C"][h],
                name=f"HubCap_{h}_{t}"
            )

    # --------------------------------------------------------
    # Factibilidad de monitoreo y reserva post-misión
    # --------------------------------------------------------

    for d in D:
        for g in G:
            for t in Tm:
                model.addConstr(
                    z[d, g, t] <= data["flyOK"][(d, g, t)],
                    name=f"FlyOK_{d}_{g}_{t}"
                )

                model.addConstr(
                    b[d, t] - data["energy"][(d, g, t)] * z[d, g, t]
                    >= data["battery_reserve"] * z[d, g, t],
                    name=f"ReserveAfterMission_{d}_{g}_{t}"
                )

    # --------------------------------------------------------
    # Dinámica de batería
    # --------------------------------------------------------

    for d in D:
        for t in Tm:
            model.addConstr(
                b[d, t + 1]
                == b[d, t]
                - gp.quicksum(data["energy"][(d, g, t)] * z[d, g, t] for g in G)
                + data["rho"][(d, t)] * q[d, t],
                name=f"BattDyn_{d}_{t}"
            )

    for d in D:
        for t in T:
            model.addConstr(b[d, t] >= data["Bmin"][d], name=f"BattMin_{d}_{t}")
            model.addConstr(b[d, t] <= data["Bmax"][d], name=f"BattMax_{d}_{t}")
            model.addConstr(
                batt_def[d, t] >= data["battery_target"] - b[d, t],
                name=f"BattDef_{d}_{t}"
            )

    # --------------------------------------------------------
    # Relación entre salidas y monitoreo de sector
    # --------------------------------------------------------

    for g in G:
        for t in Tm:
            total_monitor = gp.quicksum(z[d, g, t] for d in D)

            model.addConstr(y[g, t] <= total_monitor, name=f"YUp_{g}_{t}")
            model.addConstr(total_monitor <= len(D) * y[g, t], name=f"YLow_{g}_{t}")

    # --------------------------------------------------------
    # Penalización por urgencia y deuda operacional no atendida
    # --------------------------------------------------------

    for g in G:
        for t in Tm:
            if data["urgent"][(g, t)] == 1 and data["coverage_count"][(g, t)] > 0:
                model.addConstr(
                    miss_urgent[g, t] >= 1 - y[g, t],
                    name=f"UrgentMiss_{g}_{t}"
                )
            else:
                model.addConstr(miss_urgent[g, t] == 0, name=f"NoUrgentMiss_{g}_{t}")

            if data["deuda_operacional"][(g, t)] == 1 and data["coverage_count"][(g, t)] > 0:
                model.addConstr(
                    miss_deuda[g, t] >= 1 - y[g, t],
                    name=f"DebtMiss_{g}_{t}"
                )
            else:
                model.addConstr(miss_deuda[g, t] == 0, name=f"NoDebtMiss_{g}_{t}")

    # --------------------------------------------------------
    # Dinámica de tiempo sin monitoreo
    # --------------------------------------------------------

    M, Delta = data["BigM"], data["Delta"]

    for g in G:
        for t in Tm:
            model.addConstr(
                age[g, t + 1] <= M * (1 - y[g, t]),
                name=f"AgeReset_{g}_{t}"
            )

            model.addConstr(
                age[g, t + 1] >= age[g, t] + Delta - M * y[g, t],
                name=f"AgeIncLow_{g}_{t}"
            )

            model.addConstr(
                age[g, t + 1] <= age[g, t] + Delta + M * y[g, t],
                name=f"AgeIncUp_{g}_{t}"
            )

    # --------------------------------------------------------
    # SLA técnico
    # --------------------------------------------------------

    for g in G:
        for t in T:
            model.addConstr(
                age[g, t] <= data["L"][g] + s[g, t],
                name=f"SLA_{g}_{t}"
            )

    model.optimize()

    if model.Status == GRB.TIME_LIMIT and model.SolCount == 0:
        raise RuntimeError("Gurobi alcanzó el límite de tiempo sin solución factible.")

    if model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        raise RuntimeError(f"Modelo no resuelto correctamente. Status Gurobi: {model.Status}")

    # --------------------------------------------------------
    # Extracción de resultados
    # --------------------------------------------------------

    result = {
        "obj": model.ObjVal,
        "actions": [],
        "z": {},
        "q": {},
        "y": {},
        "b": {},
        "age": {},
        "s": {},
        "miss_urgent": {},
        "miss_deuda": {},
    }

    for d in D:
        for g in G:
            for t in Tm:
                result["z"][(d, g, t)] = z[d, g, t].X

        for t in Tm:
            result["q"][(d, t)] = q[d, t].X

        for t in T:
            result["b"][(d, t)] = b[d, t].X

    for g in G:
        for t in Tm:
            result["y"][(g, t)] = y[g, t].X
            result["miss_urgent"][(g, t)] = miss_urgent[g, t].X
            result["miss_deuda"][(g, t)] = miss_deuda[g, t].X

        for t in T:
            result["age"][(g, t)] = age[g, t].X
            result["s"][(g, t)] = s[g, t].X

    # Acción ejecutada: primera decisión del horizonte.
    for d in D:
        h = data["hub_base"][d]

        accion = {
            "dron": d,
            "tipo": "espera",
            "desde": h,
            "hacia": h,
            "retorno": h,
            "sector": None,
            "carga_en": None
        }

        for g in G:
            if result["z"][(d, g, 0)] > 0.5:
                accion = {
                    "dron": d,
                    "tipo": "monitoreo",
                    "desde": h,
                    "hacia": g,
                    "retorno": h,
                    "sector": g,
                    "carga_en": None
                }
                break

        if result["q"][(d, 0)] > 0.5:
            accion = {
                "dron": d,
                "tipo": "carga",
                "desde": h,
                "hacia": h,
                "retorno": h,
                "sector": None,
                "carga_en": h
            }

        result["actions"].append(accion)

    return result
