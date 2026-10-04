# Nodos, matriz de distancias y estado inicial de la simulación.

import math

import numpy as np
import pandas as pd


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def construir_nodos(sectores_df, hubs_df):
    sector_nodes = sectores_df.rename(columns={"sector_id": "node_id"}).copy()
    sector_nodes["node_type"] = "sector"
    hub_nodes = hubs_df.rename(columns={"hub_id": "node_id"}).copy()
    hub_nodes["node_type"] = "hub"
    hub_nodes["riesgo_hist"] = np.nan
    hub_nodes["vulnerabilidad"] = np.nan
    hub_nodes["comuna"] = "Estación de carga"
    cols = ["node_id", "node_type", "nombre", "comuna", "lat", "lon", "riesgo_hist", "vulnerabilidad"]
    return pd.concat([hub_nodes[cols], sector_nodes[cols]], ignore_index=True)


def construir_distancias(nodes_df):
    rows = []
    for _, a in nodes_df.iterrows():
        for _, b in nodes_df.iterrows():
            if a["node_id"] == b["node_id"]:
                continue
            rows.append({
                "i": a["node_id"],
                "j": b["node_id"],
                "dist_km": haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])
            })
    return pd.DataFrame(rows)


def crear_estado_inicial(drones_df, sectores_df):
    estado = {"hub": {}, "pos": {}, "bateria": {}, "age": {}, "vuelos_consecutivos": {}, "accion_previa": {}}
    for _, d in drones_df.iterrows():
        estado["hub"][d["drone_id"]] = d["hub_base"]
        estado["pos"][d["drone_id"]] = d["hub_base"]
        estado["bateria"][d["drone_id"]] = float(d["bateria_inicial"])
        estado["vuelos_consecutivos"][d["drone_id"]] = 0
        estado["accion_previa"][d["drone_id"]] = "inicial"

    # Age inicial acotado para no partir con todos los sectores vencidos,
    # pero suficientemente heterogéneo para activar priorización desde k=0.
    for _, s in sectores_df.iterrows():
        estado["age"][s["sector_id"]] = float(25 + 25*s["riesgo_hist"])
    return estado
