# Simulación por horizonte rodante: resuelve cada ciclo, actualiza el estado y genera eventos.

import copy

import numpy as np

from ecoalert.config import CONFIG
from ecoalert.riesgo import cargar_riesgo_conaf, construir_sectores
from ecoalert.flota import definir_estaciones_carga, definir_flota
from ecoalert.red import construir_nodos, construir_distancias, crear_estado_inicial
from ecoalert.prioridad import construir_data_k
from ecoalert.solver import resolver_ecoa_alert


def construir_escenario():
    conaf_riesgo_df = cargar_riesgo_conaf()

    sectores_df, capa_riesgo = construir_sectores(conaf_riesgo_df)

    sectores_df, hubs_df = definir_estaciones_carga(
        sectores_df,
        n_hubs=CONFIG["N_HUBS"],
        seed=CONFIG["SEED"],
        usar_zonas_operacionales=CONFIG["USAR_ZONAS_OPERACIONALES_COBERTURA"]
    )

    assert len(hubs_df) == CONFIG["N_HUBS"], "Se deben usar 4 estaciones de carga."

    drones_df = definir_flota(hubs_df)

    nodes_df = construir_nodos(sectores_df, hubs_df)
    dist_df = construir_distancias(nodes_df)

    return {
        "conaf_riesgo_df": conaf_riesgo_df,
        "capa_riesgo": capa_riesgo,
        "sectores_df": sectores_df,
        "hubs_df": hubs_df,
        "drones_df": drones_df,
        "nodes_df": nodes_df,
        "dist_df": dist_df,
    }


def actualizar_estado(data, estado, result):
    nuevo = copy.deepcopy(estado)

    # Los drones retornan a su estación base después de cada salida operacional.
    # Además se actualiza uso consecutivo para modelar rotación operacional.
    acciones_por_dron = {a["dron"]: a for a in result["actions"]}
    for d in data["D"]:
        h = data["hub_base"][d]
        accion = acciones_por_dron.get(d, {"tipo": "espera"})
        tipo = accion["tipo"]
        nuevo["hub"][d] = h
        nuevo["pos"][d] = h
        nuevo["bateria"][d] = float(result["b"][(d, 1)])
        nuevo["accion_previa"][d] = tipo

        if tipo == "monitoreo":
            nuevo["vuelos_consecutivos"][d] = int(nuevo.get("vuelos_consecutivos", {}).get(d, 0)) + 1
        elif tipo == "carga":
            nuevo["vuelos_consecutivos"][d] = 0
        else:
            # Espera operacional reduce fatiga/uso consecutivo sin recargar batería.
            nuevo["vuelos_consecutivos"][d] = max(0, int(nuevo.get("vuelos_consecutivos", {}).get(d, 0)) - 1)

    for g in data["G"]:
        nuevo["age"][g] = float(result["age"][(g, 1)])

    return nuevo


def simular_eventos_incendio(data, result, seed=11):
    rng = np.random.default_rng(seed + data["k"])
    eventos = []
    for g in data["G"]:
        monitoreado = result["y"][(g, 0)] > 0.5
        prioridad = data["w"][(g, 0)]
        if monitoreado:
            prob = min(0.025 + 0.10*prioridad, 0.18)
            if rng.random() < prob:
                eventos.append({"sector_id": g, "tipo_evento": "indicio_incendio", "prob_usada": prob})
    return eventos


def simular_horizonte_rodante(config, drones_df, sectores_df, hubs_df, nodes_df, dist_df, clima_df, progreso=None):
    estado = crear_estado_inicial(drones_df, sectores_df)
    snapshots = []

    for k in range(config["K_SIM"]):
        data_k = construir_data_k(k, estado, sectores_df, hubs_df, drones_df, nodes_df, dist_df, clima_df, config)
        result = resolver_ecoa_alert(data_k, verbose=False)
        eventos = simular_eventos_incendio(data_k, result, seed=config["SEED"])

        estado_pre = copy.deepcopy(estado)
        estado = actualizar_estado(data_k, estado, result)
        estado_post = copy.deepcopy(estado)

        snapshot = {
            "k": k,
            "datetime": data_k["datetime"],
            "data": data_k,
            "result": result,
            "eventos": eventos,
            "estado_pre": estado_pre,
            "estado_post": estado_post
        }
        snapshots.append(snapshot)

        if progreso is not None:
            progreso(k + 1, config["K_SIM"])

    return snapshots
