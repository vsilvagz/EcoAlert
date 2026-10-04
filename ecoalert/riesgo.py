# Riesgo histórico comunal desde registros CONAF y derivación de sectores desde el mapa de riesgo.

import re
import time
import unicodedata

import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from shapely.geometry import Point
from sklearn.cluster import KMeans

from ecoalert.config import CONFIG, CONFIG_MAPA, CONAF_RIESGO_PATH, MAPA_RIESGO_PATH

# Respaldo offline para asignar comuna (centro urbano aproximado).
COMUNAS_REF = pd.DataFrame([
    ("Valparaíso", -33.046, -71.620), ("Viña del Mar", -33.024, -71.552),
    ("Concón", -32.925, -71.515),     ("Quilpué", -33.047, -71.442),
    ("Villa Alemana", -33.042, -71.373), ("Limache", -33.010, -71.267),
    ("Olmué", -32.996, -71.187),      ("Quillota", -32.880, -71.249),
    ("La Cruz", -32.826, -71.234),    ("Calera", -32.787, -71.207),
    ("Hijuelas", -32.798, -71.144),   ("Nogales", -32.735, -71.216),
    ("Quintero", -32.782, -71.531),   ("Puchuncaví", -32.726, -71.415),
    ("Casablanca", -33.319, -71.409), ("Algarrobo", -33.368, -71.668),
], columns=["comuna", "lat", "lon"])

VULNERABILIDAD_COMUNA_REF = {
    "VALPARAISO": 0.81, "VINA DEL MAR": 0.85, "QUILPUE": 0.76, "VILLA ALEMANA": 0.70,
    "LIMACHE": 0.72, "OLMUE": 0.75, "CONCON": 0.65, "QUINTERO": 0.62,
    "PUCHUNCAVI": 0.64, "CASABLANCA": 0.76,
}

# Nombres alternativos que aparecen en CONAF u OpenStreetMap.
ALIAS_COMUNA = {
    "LA CALERA": "CALERA", "CON-CON": "CONCON", "CON CON": "CONCON",
    "LLAY-LLAY": "LLAILLAY",
}


def normalizar_texto(s):
    """Normaliza texto para cruces por comuna."""
    s = str(s).strip().upper()
    s = "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )
    s = re.sub(r"\s+", " ", s)
    return s


def cargar_riesgo_conaf(path=CONAF_RIESGO_PATH):
    conaf_riesgo_df = pd.read_csv(path)

    conaf_riesgo_df["comuna_key"] = conaf_riesgo_df["comuna_key"].apply(normalizar_texto)

    conaf_riesgo_df = conaf_riesgo_df.sort_values(
        "riesgo_hist_conaf",
        ascending=False
    ).reset_index(drop=True)

    return conaf_riesgo_df


def consolidar_conaf(conaf_riesgo_df):
    """Une comunas duplicadas por ortografía y recalcula los índices."""
    df = conaf_riesgo_df.copy()
    df["comuna_key"] = df["comuna_key"].apply(normalizar_texto).replace(ALIAS_COMUNA)
    df = (
        df.groupby("comuna_key", as_index=False)
          .agg(n_incendios_total=("n_incendios_total", "sum"),
               sup_afectada_total_ha=("sup_afectada_total_ha", "sum"),
               temporadas_con_registro=("temporadas_con_registro", "max"))
    )
    df["ocurrencia_norm"] = df["n_incendios_total"] / df["n_incendios_total"].max()
    df["dano_norm"] = df["sup_afectada_total_ha"] / df["sup_afectada_total_ha"].max()
    df["riesgo_hist_conaf"] = 0.5 * df["ocurrencia_norm"] + 0.5 * df["dano_norm"]
    return df


def cargar_capa_riesgo(path, bbox, crs_metrico, score_map):
    capa = gpd.read_file(path).set_index("Id")
    capa = capa.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]].to_crs(crs_metrico).copy()
    capa["score"] = capa["Grid code"].map(score_map).fillna(0.0)
    capa["area_km2"] = capa.geometry.area / 1e6
    return capa


def derivar_sectores_desde_mapa(capa, n_sectores, seed):
    """K-means ponderado sobre celdas Media/Alta, anclado a la celda más peligrosa."""
    cand = capa[capa["Peligrosidad"].isin(["Media", "Alta"])].copy()
    cand["px"] = cand.geometry.centroid.x
    cand["py"] = cand.geometry.centroid.y
    pesos = (cand["area_km2"].clip(upper=5.0) * cand["score"]).values

    km = KMeans(n_clusters=n_sectores, random_state=seed, n_init=20)
    cand["cluster"] = km.fit_predict(cand[["px", "py"]].values, sample_weight=pesos)

    filas = []
    for c in range(n_sectores):
        grupo = cand[cand["cluster"] == c].copy()
        cx, cy = km.cluster_centers_[c]
        grupo["d_centro"] = np.hypot(grupo["px"] - cx, grupo["py"] - cy)
        ancla = grupo.sort_values(["score", "d_centro"], ascending=[False, True]).iloc[0]
        filas.append({
            "cluster": c,
            "n_celdas_cluster": int(len(grupo)),
            "km2_alta_cluster": float(grupo.loc[grupo["Peligrosidad"] == "Alta", "area_km2"].sum()),
            "km2_media_cluster": float(grupo.loc[grupo["Peligrosidad"] == "Media", "area_km2"].sum()),
            "geometry": Point(ancla["px"], ancla["py"]),
        })
    return gpd.GeoDataFrame(filas, geometry="geometry", crs=capa.crs)


def indice_riesgo_mapa(punto, capa, radio_m):
    """Promedio ponderado por área del score en el radio; sin polígono = 0."""
    buf = punto.buffer(radio_m)
    inter = capa[capa.intersects(buf)]
    a = inter.geometry.intersection(buf).area
    return float((a * inter["score"]).sum() / buf.area), float(a.sum() / buf.area)


def comuna_por_geocodificacion(lat, lon):
    r = requests.get(
        "https://nominatim.openstreetmap.org/reverse",
        params={"lat": lat, "lon": lon, "format": "jsonv2", "zoom": 10},
        headers={"User-Agent": "EcoAlert"},
        timeout=20,
    )
    r.raise_for_status()
    ad = r.json().get("address", {})
    for k in ("municipality", "city", "town", "village"):
        if ad.get(k):
            return ad[k]
    return None


def comuna_mas_cercana(lat, lon):
    d = np.hypot(COMUNAS_REF["lat"] - lat, (COMUNAS_REF["lon"] - lon) * np.cos(np.radians(lat)))
    return COMUNAS_REF.loc[d.idxmin(), "comuna"]


def asignar_comuna(sectores_gdf, usar_geocodificacion):
    comunas = []
    for _, r in sectores_gdf.iterrows():
        comuna = None
        if usar_geocodificacion:
            try:
                comuna = comuna_por_geocodificacion(r["lat"], r["lon"])
                time.sleep(1.1)
            except Exception:
                comuna = None
        comunas.append(comuna or comuna_mas_cercana(r["lat"], r["lon"]))
    return comunas


def construir_sectores(conaf_riesgo_df):
    capa_riesgo = cargar_capa_riesgo(
        MAPA_RIESGO_PATH, CONFIG_MAPA["BBOX_OPERATIVO"],
        CONFIG_MAPA["CRS_METRICO"], CONFIG_MAPA["SCORE_GRID_CODE"],
    )

    sectores_mapa = derivar_sectores_desde_mapa(capa_riesgo, CONFIG_MAPA["N_SECTORES"], CONFIG["SEED"])

    vals = sectores_mapa.geometry.apply(
        lambda p: pd.Series(indice_riesgo_mapa(p, capa_riesgo, CONFIG_MAPA["RADIO_OPERATIVO_M"]))
    )
    sectores_mapa["riesgo_mapa"] = vals[0]
    sectores_mapa["frac_con_dato_mapa"] = vals[1]

    ll = sectores_mapa.to_crs(4326)
    sectores_mapa["lat"] = ll.geometry.y.round(4)
    sectores_mapa["lon"] = ll.geometry.x.round(4)

    sectores_mapa = sectores_mapa.sort_values("riesgo_mapa", ascending=False).reset_index(drop=True)
    sectores_mapa["sector_id"] = [f"M{i + 1:02d}" for i in range(len(sectores_mapa))]
    sectores_mapa["comuna"] = asignar_comuna(sectores_mapa, CONFIG_MAPA["USAR_GEOCODIFICACION"])
    sectores_mapa["comuna_key"] = sectores_mapa["comuna"].apply(normalizar_texto).replace(ALIAS_COMUNA)

    rep = sectores_mapa.groupby("comuna").cumcount() + 1
    tot = sectores_mapa.groupby("comuna")["sector_id"].transform("count")
    sectores_mapa["nombre"] = np.where(
        tot > 1,
        sectores_mapa["comuna"] + " - núcleo " + rep.astype(str),
        sectores_mapa["comuna"] + " - núcleo de riesgo",
    )

    # Riesgo comunal CONAF consolidado + mezcla con el índice del mapa.
    conaf_consolidado_df = consolidar_conaf(conaf_riesgo_df)
    sectores_mapa = sectores_mapa.merge(
        conaf_consolidado_df[["comuna_key", "n_incendios_total", "sup_afectada_total_ha",
                              "ocurrencia_norm", "dano_norm", "riesgo_hist_conaf"]],
        on="comuna_key", how="left",
    )
    sectores_mapa["riesgo_hist_conaf"] = sectores_mapa["riesgo_hist_conaf"].fillna(0.0)
    sectores_mapa["riesgo_hist_base"] = sectores_mapa["riesgo_mapa"]
    sectores_mapa["riesgo_hist"] = (
        CONFIG_MAPA["PESO_MAPA"] * sectores_mapa["riesgo_mapa"]
        + CONFIG_MAPA["PESO_COMUNAL"] * sectores_mapa["riesgo_hist_conaf"]
    ).clip(0, 1)

    sectores_mapa["vulnerabilidad"] = (
        sectores_mapa["comuna_key"].map(VULNERABILIDAD_COMUNA_REF)
        .fillna(CONFIG_MAPA["VULNERABILIDAD_DEFAULT"]).round(2)
    )

    sectores_df = pd.DataFrame(sectores_mapa.drop(columns="geometry"))[[
        "sector_id", "nombre", "comuna", "lat", "lon",
        "riesgo_hist_base", "vulnerabilidad", "comuna_key",
        "n_incendios_total", "sup_afectada_total_ha", "ocurrencia_norm", "dano_norm",
        "riesgo_hist_conaf", "riesgo_mapa", "frac_con_dato_mapa", "riesgo_hist",
        "n_celdas_cluster", "km2_alta_cluster", "km2_media_cluster",
    ]]

    return sectores_df, capa_riesgo
