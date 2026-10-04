# Definición de estaciones de carga fijas y de la flota de drones simulada.

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from ecoalert.config import CONFIG


def definir_estaciones_carga(sectores_df, n_hubs=4, seed=11, usar_zonas_operacionales=True):
    """
    Define cuatro estaciones de carga fijas para la simulación.

    Las estaciones representan puntos de operación y recarga para la flota simulada.
    Su ubicación se define con criterio territorial, considerando la distribución
    de sectores, el riesgo histórico comunal estimado desde registros CONAF y los
    parámetros operacionales de los drones.

    En una implementación futura, la localización de estaciones podría optimizarse
    con datos reales de infraestructura, bases operativas, autonomía de la flota
    disponible y restricciones logísticas de la institución usuaria.
    """

    sectores_out = sectores_df.copy()

    if usar_zonas_operacionales:
        zonas = {
            0: {
                "hub_id": "H1",
                "nombre": "Estación norte costa-industrial",
                "sectores": ["S07", "S08", "S09"],
                "zona_operacional": "Concón, Quintero y Puchuncaví",
                "lat_ref": -32.820,
                "lon_ref": -71.480,
            },
            1: {
                "hub_id": "H2",
                "nombre": "Estación centro urbano-interior",
                "sectores": ["S02", "S03", "S04"],
                "zona_operacional": "Viña del Mar, Quilpué y Villa Alemana",
                "lat_ref": -33.035,
                "lon_ref": -71.440,
            },
            2: {
                "hub_id": "H3",
                "nombre": "Estación interior norte",
                "sectores": ["S05", "S06"],
                "zona_operacional": "Limache y Olmué",
                "lat_ref": -33.005,
                "lon_ref": -71.225,
            },
            3: {
                "hub_id": "H4",
                "nombre": "Estación sur-interior",
                "sectores": ["S01", "S10", "S11", "S12"],
                "zona_operacional": "Valparaíso, Peñuelas, Casablanca y Colliguay",
                "lat_ref": -33.205,
                "lon_ref": -71.420,
            },
        }

        asignacion = {}
        for zona_id, zinfo in zonas.items():
            for g in zinfo["sectores"]:
                asignacion[g] = zona_id

        sectores_out["hub_asociado_cluster"] = sectores_out["sector_id"].map(asignacion)

        faltantes = sectores_out["hub_asociado_cluster"].isna()
        if faltantes.any():
            coords = sectores_out.loc[faltantes, ["lat", "lon"]].values
            if len(coords) > 0:
                kmeans = KMeans(
                    n_clusters=min(n_hubs, len(coords)),
                    random_state=seed,
                    n_init=20
                )
                labels = kmeans.fit_predict(coords)
                sectores_out.loc[faltantes, "hub_asociado_cluster"] = labels + len(zonas)

        hubs = []
        for zona_id in sorted(sectores_out["hub_asociado_cluster"].dropna().unique()):
            zona_id = int(zona_id)
            grupo = sectores_out[sectores_out["hub_asociado_cluster"] == zona_id]
            zinfo = zonas.get(zona_id, {})

            # La posición de cada estación combina un punto operativo de referencia
            # con el centroide ponderado por riesgo histórico de los sectores asignados.
            # Esto permite representar infraestructura fija con cobertura territorial
            # sin resolver todavía un problema completo de localización óptima.
            w = grupo["riesgo_hist"].clip(lower=0.05).values
            lat_centroide = np.average(grupo["lat"].values, weights=w)
            lon_centroide = np.average(grupo["lon"].values, weights=w)

            lat_ref = zinfo.get("lat_ref", lat_centroide)
            lon_ref = zinfo.get("lon_ref", lon_centroide)

            lat_h = 0.70 * lat_ref + 0.30 * lat_centroide
            lon_h = 0.70 * lon_ref + 0.30 * lon_centroide

            hubs.append({
                "hub_id": zinfo.get("hub_id", f"H{zona_id + 1}"),
                "nombre": zinfo.get("nombre", f"Estación de carga zona {zona_id + 1}"),
                "zona_operacional": zinfo.get("zona_operacional", "Zona operativa"),
                "lat": float(lat_h),
                "lon": float(lon_h),
                "capacidad": 2,
                "carga_por_hora": CONFIG.get("CARGA_POR_HORA_DEFAULT", 38.0)
            })

        hubs_df = pd.DataFrame(hubs).sort_values("hub_id").reset_index(drop=True)
        return sectores_out, hubs_df

    coords = sectores_df[["lat", "lon"]].values
    weights = sectores_df["riesgo_hist"].values

    kmeans = KMeans(n_clusters=n_hubs, random_state=seed, n_init=20)
    kmeans.fit(coords, sample_weight=weights)

    sectores_out["hub_asociado_cluster"] = kmeans.labels_

    hubs = []
    for cluster in sorted(sectores_out["hub_asociado_cluster"].unique()):
        grupo = sectores_out[sectores_out["hub_asociado_cluster"] == cluster]
        w = grupo["riesgo_hist"].clip(lower=0.05).values

        lat_h = np.average(grupo["lat"].values, weights=w)
        lon_h = np.average(grupo["lon"].values, weights=w)

        hubs.append({
            "hub_id": f"H{cluster + 1}",
            "nombre": f"Estación de carga zona {cluster + 1}",
            "zona_operacional": "Agrupamiento ponderado por riesgo histórico",
            "lat": float(lat_h),
            "lon": float(lon_h),
            "capacidad": 2,
            "carga_por_hora": CONFIG.get("CARGA_POR_HORA_DEFAULT", 38.0)
        })

    hubs_df = pd.DataFrame(hubs).sort_values("hub_id").reset_index(drop=True)
    return sectores_out, hubs_df


def definir_flota(hubs_df):
    # Flota simulada de 6 drones. La asignación por estación considera
    # cobertura territorial, dispersión de sectores y necesidad de alternar entre
    # monitoreo, carga y espera operacional.

    # Los IDs de las estaciones (hub_base) coinciden con los generados
    # dinámicamente al definir las estaciones de carga (H5, H6, H7, H8).

    drones_df = pd.DataFrame([
        # H5: norte costa-industrial
        {
            "drone_id": "D1",
            "tipo": "Estándar",
            "velocidad_kmh": 45,
            "autonomia_min": 50,
            "bateria_max": 100,
            "bateria_min": 12,
            "bateria_inicial": 90,
            "consumo_base_por_km": 1.05,
            "hub_base": "H5"
        },

        # H6: eje urbano-interior
        {
            "drone_id": "D2",
            "tipo": "Estándar",
            "velocidad_kmh": 45,
            "autonomia_min": 50,
            "bateria_max": 100,
            "bateria_min": 12,
            "bateria_inicial": 86,
            "consumo_base_por_km": 1.05,
            "hub_base": "H6"
        },
        {
            "drone_id": "D3",
            "tipo": "Mayor autonomía",
            "velocidad_kmh": 54,
            "autonomia_min": 65,
            "bateria_max": 100,
            "bateria_min": 12,
            "bateria_inicial": 78,
            "consumo_base_por_km": 0.88,
            "hub_base": "H6"
        },

        # H7: interior norte
        {
            "drone_id": "D4",
            "tipo": "Mayor autonomía",
            "velocidad_kmh": 54,
            "autonomia_min": 65,
            "bateria_max": 100,
            "bateria_min": 12,
            "bateria_inicial": 84,
            "consumo_base_por_km": 0.88,
            "hub_base": "H7"
        },

        # H8: sur-interior
        {
            "drone_id": "D5",
            "tipo": "Rápido",
            "velocidad_kmh": 62,
            "autonomia_min": 55,
            "bateria_max": 100,
            "bateria_min": 12,
            "bateria_inicial": 88,
            "consumo_base_por_km": 1.18,
            "hub_base": "H8"
        },
        {
            "drone_id": "D6",
            "tipo": "Rápido",
            "velocidad_kmh": 62,
            "autonomia_min": 55,
            "bateria_max": 100,
            "bateria_min": 12,
            "bateria_inicial": 76,
            "consumo_base_por_km": 1.18,
            "hub_base": "H8"
        },
    ])

    assert len(drones_df) == 6, "La flota simulada debe tener 6 drones."
    assert set(drones_df["hub_base"]) <= set(hubs_df["hub_id"]), "Hay drones asignados a estaciones no definidas."
    assert set(hubs_df["hub_id"]) <= set(drones_df["hub_base"]), "Toda estación debe tener al menos un dron asignado."

    return drones_df
