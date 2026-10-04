# Tablas de resultados, métricas operativas y exportación a CSV.

import os
import pickle

import numpy as np
import pandas as pd

from ecoalert.config import CONFIG


def construir_tablas_resultados(snapshots):
    acciones_rows, drones_rows, sectores_rows, eventos_rows = [], [], [], []

    for snap in snapshots:
        k, dt, data, result, estado_post = snap["k"], snap["datetime"], snap["data"], snap["result"], snap["estado_post"]

        for a in result["actions"]:
            acciones_rows.append({
                "k": k,
                "datetime": dt,
                "dron": a["dron"],
                "accion": a["tipo"],
                "desde": a["desde"],
                "sector": a["sector"],
                "retorno": a["retorno"],
                "carga_en": a["carga_en"],
                "bateria_post_%": estado_post["bateria"][a["dron"]],
            })

        for d in data["D"]:
            drones_rows.append({
                "k": k,
                "datetime": dt,
                "dron": d,
                "hub_base": data["hub_base"][d],
                "bateria_%": estado_post["bateria"][d]
            })

        for g in data["G"]:
            comps = data.get("priority_components", {}).get((g, 0), {})
            clima_g = data["clima_node"][(g, 0)]
            sectores_rows.append({
                "k": k,
                "datetime": dt,
                "sector": g,
                "age_pre_min": data["age0"][g],
                "age_post_min": estado_post["age"][g],
                "sla_min": data["L"][g],
                "monitoreado": result["y"][(g, 0)] > 0.5,
                "prioridad": data["w"][(g, 0)],
                "prioritario": data["critical"][(g, 0)] == 1,
                "urgente_operacional": data.get("urgent", {}).get((g, 0), 0) == 1,
                "drones_factibles": data.get("coverage_count", {}).get((g, 0), np.nan),
                "riesgo_hist": comps.get("riesgo_hist", np.nan),
                "riesgo_climatico": comps.get("riesgo_climatico", clima_g["riesgo_climatico"]),
                "age_norm": comps.get("age_norm", np.nan),
                "vulnerabilidad": comps.get("vulnerabilidad", np.nan),
                "comp_hist": comps.get("comp_hist", np.nan),
                "comp_clima": comps.get("comp_clima", np.nan),
                "comp_age": comps.get("comp_age", np.nan),
                "comp_vuln": comps.get("comp_vuln", np.nan),
                "abandono_boost": comps.get("abandono_boost", np.nan),
                "penalizacion_reciente": comps.get("penalizacion_reciente", np.nan),
                "miss_urgent": result.get("miss_urgent", {}).get((g, 0), np.nan),
                "temperatura": clima_g["temperatura"],
                "humedad": clima_g["humedad"],
                "viento": clima_g["viento"],
                "precipitacion": clima_g["precipitacion"],
            })

        for ev in snap["eventos"]:
            eventos_rows.append({"k": k, "datetime": dt, **ev})

    return (
        pd.DataFrame(acciones_rows),
        pd.DataFrame(drones_rows),
        pd.DataFrame(sectores_rows),
        pd.DataFrame(eventos_rows),
    )


def calcular_metricas(snapshots):
    rows = []
    for snap in snapshots:
        k, dt, data, result, estado_post = snap["k"], snap["datetime"], snap["data"], snap["result"], snap["estado_post"]
        D, G = data["D"], data["G"]
        acciones = result["actions"]

        monitoreados = [g for g in G if result["y"][(g, 0)] > 0.5]
        prioritarios = [g for g in G if data["w"][(g, 0)] >= data["priority_threshold"]]
        prioritarios_mon = [g for g in monitoreados if g in prioritarios]

        total_priority_mass = sum(data["w"][(g, 0)] for g in prioritarios)
        monitored_priority_mass = sum(data["w"][(g, 0)] for g in prioritarios_mon)
        cobertura_prioritaria = monitored_priority_mass / total_priority_mass if total_priority_mass > 0 else 1.0

        vuelos = sum(1 for a in acciones if a["tipo"] == "monitoreo")
        cargas = sum(1 for a in acciones if a["tipo"] == "carga")
        esperas = sum(1 for a in acciones if a["tipo"] == "espera")

        distancia_total = 0.0
        consumo_total = 0.0
        for a in acciones:
            if a["tipo"] == "monitoreo":
                d, g = a["dron"], a["sector"]
                distancia_total += data["sortie_dist"][(d, g, 0)]
                consumo_total += data["energy"][(d, g, 0)]

        bat_vals = [estado_post["bateria"][d] for d in D]
        age_vals = [estado_post["age"][g] for g in G]
        sectores_sobre_sla = [g for g in G if estado_post["age"][g] > data["L"][g]]
        pendientes_criticos = [g for g in prioritarios if g not in prioritarios_mon]

        rows.append({
            "k": k,
            "datetime": dt,
            "cobertura_prioritaria": cobertura_prioritaria,
            "prioritarios_total": len(prioritarios),
            "prioritarios_monitoreados": len(prioritarios_mon),
            "pendientes_criticos": len(pendientes_criticos),
            "sectores_sobre_SLA": len(sectores_sobre_sla),
            "sectores_monitoreados": len(monitoreados),
            "vuelos": vuelos,
            "cargas": cargas,
            "esperas": esperas,
            "bateria_promedio": np.mean(bat_vals),
            "bateria_minima": np.min(bat_vals),
            "age_promedio": np.mean(age_vals),
            "age_maximo": np.max(age_vals),
            "distancia_total_km": distancia_total,
            "consumo_estimado_pct": consumo_total,
            "drones_sobre_70": sum(1 for v in bat_vals if v >= 70),
            "drones_bajo_45": sum(1 for v in bat_vals if v < 45),
            "eventos": len(snap["eventos"])
        })
    return pd.DataFrame(rows)


def consolidar_resultados(snapshots, escenario, clima_df, config):
    acciones_df, drones_estado_df, sectores_estado_df, eventos_df = construir_tablas_resultados(snapshots)
    metricas_df = calcular_metricas(snapshots)

    return {
        "config": config,
        "snapshots": snapshots,
        "clima_df": clima_df,
        "sectores_df": escenario["sectores_df"],
        "hubs_df": escenario["hubs_df"],
        "drones_df": escenario["drones_df"],
        "conaf_riesgo_df": escenario["conaf_riesgo_df"],
        "acciones_df": acciones_df,
        "drones_estado_df": drones_estado_df,
        "sectores_estado_df": sectores_estado_df,
        "eventos_df": eventos_df,
        "metricas_df": metricas_df,
    }


def exportar_resultados(res):
    RESULTS_DIR = CONFIG["OUTPUT_DIR"]
    CSV_DIR = os.path.join(RESULTS_DIR, "01_csv")

    os.makedirs(CSV_DIR, exist_ok=True)

    # ------------------------------------------------------------
    # Exportar resultados principales
    # ------------------------------------------------------------

    res["acciones_df"].to_csv(os.path.join(CSV_DIR, "acciones_drones.csv"), index=False)
    res["drones_estado_df"].to_csv(os.path.join(CSV_DIR, "estado_drones_por_k.csv"), index=False)
    res["sectores_estado_df"].to_csv(os.path.join(CSV_DIR, "estado_sectores_por_k.csv"), index=False)
    res["eventos_df"].to_csv(os.path.join(CSV_DIR, "eventos_simulados.csv"), index=False)
    res["metricas_df"].to_csv(os.path.join(CSV_DIR, "metricas_operativas.csv"), index=False)

    # ------------------------------------------------------------
    # Exportar datos usados
    # ------------------------------------------------------------

    res["sectores_df"].to_csv(os.path.join(CSV_DIR, "sectores_con_riesgo_conaf.csv"), index=False)
    res["hubs_df"].to_csv(os.path.join(CSV_DIR, "estaciones_carga.csv"), index=False)
    res["drones_df"].to_csv(os.path.join(CSV_DIR, "drones_simulados.csv"), index=False)
    res["clima_df"].to_csv(os.path.join(CSV_DIR, "clima_pronostico_usado.csv"), index=False)
    res["conaf_riesgo_df"].to_csv(os.path.join(CSV_DIR, "riesgo_historico_conaf_por_comuna.csv"), index=False)

    # ------------------------------------------------------------
    # README
    # ------------------------------------------------------------

    readme_text = f"""EcoAlert — Resultados de la simulación

    Carpeta:
    resultados → 01_csv

    Contenido:
    - acciones_drones.csv: decisiones del solver por ciclo k y dron.
    - estado_drones_por_k.csv: batería y estado de cada dron por ciclo.
    - estado_sectores_por_k.csv: prioridad, clima y tiempo sin vigilancia por sector.
    - eventos_simulados.csv: indicios de incendio simulados.
    - metricas_operativas.csv: métricas operativas simuladas.
    - sectores_con_riesgo_conaf.csv: sectores con riesgo histórico asignado desde CONAF.
    - estaciones_carga.csv: estaciones de carga usadas en la simulación.
    - drones_simulados.csv: parámetros de la flota simulada.
    - clima_pronostico_usado.csv: pronóstico horario utilizado como horizonte climático.
    - riesgo_historico_conaf_por_comuna.csv: indicador territorial calculado desde registros CONAF.

    Modo climático usado: {CONFIG['CLIMATE_MODE']}

    Nota:
    La visualización principal del mapa se revisa en la app Streamlit, usando el selector de ciclo k.
    """

    with open(os.path.join(RESULTS_DIR, "README_resultados.txt"), "w", encoding="utf-8") as f:
        f.write(readme_text)

    return RESULTS_DIR


SIMULACION_PATH = os.path.join(CONFIG["OUTPUT_DIR"], "ultima_simulacion.pkl")


def guardar_simulacion(res, path=SIMULACION_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(res, f)


def cargar_simulacion(path=SIMULACION_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)
