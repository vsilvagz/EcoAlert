# Prioridad operacional dinámica y construcción de parámetros del solver por ciclo k.

import copy

import numpy as np


def calcular_boost_abandono(age_min, config):
    umbral = config["UMBRAL_ABANDONO_MIN"]
    age_norm_ref = config["AGE_NORMALIZACION_MIN"]
    boost_max = config["BOOST_ABANDONO_MAX"]

    if age_min <= umbral:
        return 0.0

    progreso = np.clip((age_min - umbral) / max(age_norm_ref - umbral, 1.0), 0, 1)
    return float(boost_max * progreso)


def calcular_prioridad_operacional(row, age_min, config):
    """Prioridad operacional dinámica usada por el solver."""

    age_norm = float(np.clip(age_min / config["AGE_NORMALIZACION_MIN"], 0, 1))
    abandono_boost = calcular_boost_abandono(age_min, config)

    penalizacion_reciente = (
        config["PENALIZACION_RECIENTE"]
        if age_min < config["VENTANA_RECIEN_MONITOREADO_MIN"]
        else 0.0
    )

    comp_hist = config["PESO_RIESGO_HIST"] * float(row["riesgo_hist"])
    comp_clima = config["PESO_RIESGO_CLIMA"] * float(row["riesgo_climatico"])
    comp_age = config["PESO_AGE"] * age_norm
    comp_vuln = config["PESO_VULNERABILIDAD"] * float(row["vulnerabilidad"])

    prioridad = comp_hist + comp_clima + comp_age + comp_vuln + abandono_boost - penalizacion_reciente
    prioridad = float(np.clip(prioridad, 0, 1.25))

    return prioridad, {
        "riesgo_hist": float(row["riesgo_hist"]),
        "riesgo_climatico": float(row["riesgo_climatico"]),
        "age_min": float(age_min),
        "age_norm": age_norm,
        "vulnerabilidad": float(row["vulnerabilidad"]),
        "comp_hist": float(comp_hist),
        "comp_clima": float(comp_clima),
        "comp_age": float(comp_age),
        "comp_vuln": float(comp_vuln),
        "abandono_boost": float(abandono_boost),
        "penalizacion_reciente": float(penalizacion_reciente),
        "prioridad": prioridad,
    }


def construir_data_k(k, estado, sectores_df, hubs_df, drones_df, nodes_df, dist_df, clima_df, config):
    times = sorted(clima_df["datetime"].unique())
    HORIZON = config["HORIZON"]

    if k + HORIZON > len(times):
        raise ValueError(f"No hay suficientes horas climáticas para k={k} con HORIZON={HORIZON}.")

    horizon_times = times[k:k + HORIZON]

    D = drones_df["drone_id"].tolist()
    G = sectores_df["sector_id"].tolist()
    H = hubs_df["hub_id"].tolist()

    T = list(range(HORIZON))
    Tm = list(range(HORIZON - 1))

    dist_lookup = {(r.i, r.j): r.dist_km for _, r in dist_df.iterrows()}

    # --------------------------------------------------------
    # SLA técnico por sector
    # --------------------------------------------------------

    L = {}

    for _, r in sectores_df.iterrows():
        sla = config["SLA_BASE_MIN"] - config["SLA_REDUCCION_RIESGO_MIN"] * float(r.riesgo_hist)
        L[r.sector_id] = float(max(config["SLA_MIN_MIN"], sla))

    # --------------------------------------------------------
    # Prioridad por sector y hora
    # --------------------------------------------------------

    w = {}
    critical = {}
    urgent = {}
    priority_components = {}

    for t_idx, dt in enumerate(horizon_times):
        clima_t = clima_df[clima_df["datetime"] == dt]

        for _, row in clima_t.iterrows():
            g = row["sector_id"]
            age_proyectado = float(estado["age"][g] + t_idx * config["DELTA_MIN"])

            prioridad, comps = calcular_prioridad_operacional(row, age_proyectado, config)

            w[(g, t_idx)] = prioridad
            priority_components[(g, t_idx)] = comps

            critical[(g, t_idx)] = 1 if prioridad >= config["PRIORITY_THRESHOLD"] else 0

            urgent[(g, t_idx)] = 1 if (
                age_proyectado >= L[g]
                or prioridad >= config["URGENT_PRIORITY_THRESHOLD"]
            ) else 0

    # --------------------------------------------------------
    # Clima por sector y por estación
    # --------------------------------------------------------

    clima_node = {}

    for t_idx, dt in enumerate(horizon_times):
        clima_t = clima_df[clima_df["datetime"] == dt]

        promedio = {
            "temperatura": float(clima_t["temperatura"].mean()),
            "humedad": float(clima_t["humedad"].mean()),
            "viento": float(clima_t["viento"].mean()),
            "precipitacion": float(clima_t["precipitacion"].mean()),
            "presion": float(clima_t["presion"].mean()),
            "riesgo_climatico": float(clima_t["riesgo_climatico"].mean())
        }

        for _, row in clima_t.iterrows():
            clima_node[(row["sector_id"], t_idx)] = {
                "temperatura": float(row["temperatura"]),
                "humedad": float(row["humedad"]),
                "viento": float(row["viento"]),
                "precipitacion": float(row["precipitacion"]),
                "presion": float(row["presion"]),
                "riesgo_climatico": float(row["riesgo_climatico"])
            }

        for h in H:
            clima_node[(h, t_idx)] = promedio

    # --------------------------------------------------------
    # Parámetros de misión: estación base -> sector -> estación base
    # --------------------------------------------------------

    flyOK = {}
    sortie_cost = {}
    sortie_time = {}
    sortie_dist = {}
    energy = {}

    coverage_count = {}
    feasible_drones = {}

    battery_penalty = {}
    recent_use_penalty = {}
    force_charge0 = {}
    force_rest0 = {}

    hubs_df_idx = hubs_df.set_index("hub_id")

    for _, drow in drones_df.iterrows():
        d = drow["drone_id"]
        h = drow["hub_base"]

        speed = float(drow["velocidad_kmh"])
        autonomy = float(drow["autonomia_min"])
        base_cons = float(drow["consumo_base_por_km"])

        b0 = float(estado["bateria"][d])
        vuelos_cons = int(estado.get("vuelos_consecutivos", {}).get(d, 0))

        # La carga se fuerza solo bajo reserva operacional crítica.
        # Entre 30% y 45%, el dron puede salir si la misión es prioritaria
        # y mantiene reserva post-misión.
        force_charge0[d] = 1 if b0 <= config["BATERIA_FORZAR_CARGA"] else 0
        force_rest0[d] = 1 if vuelos_cons >= config["MAX_VUELOS_CONSECUTIVOS"] else 0

        recent_use_penalty[d] = config["PENALIZACION_USO_RECIENTE_DRON"] * vuelos_cons

        if b0 < config["BATERIA_MIN_SALIDA_URGENTE"]:
            battery_penalty[d] = config["PENALIZACION_BATERIA_BAJA"]
        elif b0 < config["BATERIA_MIN_SALIDA_NORMAL"]:
            battery_penalty[d] = config["PENALIZACION_BATERIA_MEDIA"]
        else:
            battery_penalty[d] = 0.0

        for g in G:
            dist_one_way = dist_lookup[(h, g)]
            round_dist = 2.0 * dist_one_way

            for t in Tm:
                cg = clima_node[(g, t)]
                ch = clima_node[(h, t)]

                viento = max(cg["viento"], ch["viento"])
                precip = max(cg["precipitacion"], ch["precipitacion"])

                tiempo_min = 60.0 * round_dist / speed

                factor_viento = 1.0 + 0.018 * viento
                factor_reserva = 1.08
                consumo = base_cons * round_dist * factor_viento * factor_reserva

                prioridad_t = w[(g, t)]
                age_t = priority_components[(g, t)]["age_min"]

                salida_urgente = (
                    prioridad_t >= config["URGENT_PRIORITY_THRESHOLD"]
                    or age_t >= config["DEUDA_OPERACIONAL_MIN"]
                )

                min_salida = (
                    config["BATERIA_MIN_SALIDA_URGENTE"]
                    if salida_urgente
                    else config["BATERIA_MIN_SALIDA_NORMAL"]
                )

                ok = (
                    tiempo_min <= 0.95 * autonomy
                    and viento <= config["MAX_WIND_KMH"]
                    and precip <= config["MAX_PRECIP_MM"]
                    and b0 >= min_salida
                    and b0 - consumo >= config["BATERIA_RESERVA_POST_MISION"]
                    and force_rest0[d] == 0
                    and force_charge0[d] == 0
                )

                flyOK[(d, g, t)] = 1 if ok else 0
                sortie_dist[(d, g, t)] = float(round_dist)
                sortie_time[(d, g, t)] = float(tiempo_min)
                sortie_cost[(d, g, t)] = float(tiempo_min + recent_use_penalty[d] + battery_penalty[d])
                energy[(d, g, t)] = float(consumo)

    for g in G:
        for t in Tm:
            drones_ok = [d for d in D if flyOK[(d, g, t)] == 1]
            feasible_drones[(g, t)] = drones_ok
            coverage_count[(g, t)] = len(drones_ok)

    # --------------------------------------------------------
    # Deuda operacional
    # --------------------------------------------------------
    # La deuda operacional evita que sectores factibles queden postergados
    # indefinidamente. Se usa una lógica escalonada:
    # - alerta temprana desde 120 min solo si la prioridad ya es alta;
    # - deuda base desde 180 min;
    # - deuda crítica desde 240 min;
    # - criterio relativo al SLA técnico para sectores con exigencia mayor.

    deuda_operacional = {}

    for g in G:
        for t in Tm:
            age_t = priority_components[(g, t)]["age_min"]
            prioridad_t = w[(g, t)]

            umbral_sla = config["DEUDA_RELATIVA_SLA"] * L[g]

            deuda_alerta_temprana = (
                age_t >= config["DEUDA_ALERTA_TEMPRANA_MIN"]
                and prioridad_t >= config["DEUDA_PRIORIDAD_MIN"]
            )

            deuda_base = (
                age_t >= config["DEUDA_OPERACIONAL_MIN"]
            )

            deuda_critica = (
                age_t >= config["DEUDA_CRITICA_MIN"]
            )

            deuda_por_sla = (
                age_t >= umbral_sla
                and prioridad_t >= 0.50
            )

            deuda = (
                deuda_alerta_temprana
                or deuda_base
                or deuda_critica
                or deuda_por_sla
            )

            # Solo se activa si existe al menos un dron factible para el sector.
            # Si no hay factibilidad operacional, el solver no se fuerza artificialmente.
            deuda_operacional[(g, t)] = 1 if deuda and coverage_count[(g, t)] > 0 else 0

    # --------------------------------------------------------
    # Carga de batería
    # --------------------------------------------------------

    rho = {}

    for _, drow in drones_df.iterrows():
        d = drow["drone_id"]
        h = drow["hub_base"]

        charge_rate = float(hubs_df_idx.loc[h, "carga_por_hora"])

        for t in Tm:
            rho[(d, t)] = charge_rate

    Bmax = {r.drone_id: float(r.bateria_max) for _, r in drones_df.iterrows()}
    Bmin = {r.drone_id: float(r.bateria_min) for _, r in drones_df.iterrows()}
    hub_base = {r.drone_id: r.hub_base for _, r in drones_df.iterrows()}

    return {
        "k": k,
        "datetime": horizon_times[0],
        "horizon_times": horizon_times,
        "D": D,
        "G": G,
        "H": H,
        "T": T,
        "Tm": Tm,
        "nodes_df": nodes_df,
        "sectores_df": sectores_df,
        "hubs_df": hubs_df,
        "drones_df": drones_df,
        "clima_node": clima_node,
        "w": w,
        "critical": critical,
        "urgent": urgent,
        "deuda_operacional": deuda_operacional,
        "priority_components": priority_components,
        "flyOK": flyOK,
        "coverage_count": coverage_count,
        "feasible_drones": feasible_drones,
        "sortie_cost": sortie_cost,
        "sortie_time": sortie_time,
        "sortie_dist": sortie_dist,
        "energy": energy,
        "rho": rho,
        "Bmax": Bmax,
        "Bmin": Bmin,
        "hub_base": hub_base,
        "b0": copy.deepcopy(estado["bateria"]),
        "age0": copy.deepcopy(estado["age"]),
        "vuelos_consecutivos0": copy.deepcopy(estado.get("vuelos_consecutivos", {})),
        "accion_previa0": copy.deepcopy(estado.get("accion_previa", {})),
        "force_charge0": force_charge0,
        "force_rest0": force_rest0,
        "battery_penalty": battery_penalty,
        "recent_use_penalty": recent_use_penalty,
        "L": L,
        "C": {r.hub_id: int(r.capacidad) for _, r in hubs_df.iterrows()},
        "Delta": config["DELTA_MIN"],
        "BigM": config["BIG_M"],
        "lam": config["LAMBDA_TRAVEL"],
        "mu": config["MU_SLACK"],
        "eta_batt": config["ETA_BATTERY_DEFICIT"],
        "beta_monitor": config["BETA_MONITORING_REWARD"],
        "beta_critical": config["BETA_CRITICAL_REWARD"],
        "beta_deuda": config["BETA_DEUDA_MONITOR"],
        "gamma_urgent_miss": config["GAMMA_URGENT_MISS"],
        "gamma_deuda_miss": config["GAMMA_DEUDA_NO_ATENDIDA"],
        "battery_target": config["BATERIA_OBJETIVO_OPERACIONAL"],
        "battery_reserve": config["BATERIA_RESERVA_POST_MISION"],
        "priority_threshold": config["PRIORITY_THRESHOLD"]
    }
