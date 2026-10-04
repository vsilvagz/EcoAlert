# Pronóstico horario (Open-Meteo o simulado) y cálculo del riesgo climático por sector.

import numpy as np
import pandas as pd
import requests


def _clima_df_desde_hourly(hourly, sector):
    times = hourly.get("time", [])
    n = len(times)

    return pd.DataFrame({
        "datetime": pd.to_datetime(times),
        "sector_id": sector["sector_id"],
        "temperatura": hourly.get("temperature_2m", [np.nan] * n),
        "humedad": hourly.get("relative_humidity_2m", [np.nan] * n),
        "viento": hourly.get("wind_speed_10m", [np.nan] * n),
        "precipitacion": hourly.get("precipitation", [0] * n),
        "presion": hourly.get("surface_pressure", [np.nan] * n),
    })


def descargar_clima_openmeteo_forecast_sector(sector, forecast_days, timezone):
    """
    Consulta pronóstico horario desde Open-Meteo Forecast API para un sector.
    El resultado se utiliza como horizonte climático de planificación.
    """

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": sector["lat"],
        "longitude": sector["lon"],
        "forecast_days": forecast_days,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,surface_pressure",
        "timezone": timezone
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    js = response.json()
    return _clima_df_desde_hourly(js["hourly"], sector)


def generar_clima_simulado(sectores_df, config, seed=11):
    """
    Serie climática sintética de contingencia.

    Permite mantener ejecutable la simulación si la consulta externa de clima no
    está disponible. No reemplaza la fuente principal de pronóstico, pero conserva
    la estructura horaria requerida por el modelo.
    """

    rng = np.random.default_rng(seed)

    n_horas = config["K_SIM"] + config["HORIZON"] + 4
    start = pd.Timestamp.now(tz=config["TIMEZONE"]).floor("h").tz_localize(None)
    times = pd.date_range(start=start, periods=n_horas, freq="h")

    rows = []

    for _, s in sectores_df.iterrows():
        base_temp = rng.uniform(17, 25)
        base_hum = rng.uniform(45, 75)
        base_wind = rng.uniform(6, 18)

        for dt in times:
            hour = dt.hour

            temp = base_temp + 7 * np.sin((hour - 8) / 24 * 2 * np.pi) + rng.normal(0, 1.0)
            hum = base_hum - 14 * np.sin((hour - 8) / 24 * 2 * np.pi) + rng.normal(0, 4)
            wind = base_wind + 5 * np.sin((hour - 12) / 24 * 2 * np.pi) + rng.normal(0, 2)
            precip = max(0, rng.normal(0.12, 0.35))
            presion = 1015 + rng.normal(0, 3)

            rows.append({
                "datetime": dt,
                "sector_id": s["sector_id"],
                "temperatura": float(np.clip(temp, 5, 42)),
                "humedad": float(np.clip(hum, 10, 100)),
                "viento": float(np.clip(wind, 0, 55)),
                "precipitacion": float(np.clip(precip, 0, 8)),
                "presion": float(np.clip(presion, 980, 1035)),
            })

    return pd.DataFrame(rows)


def cargar_clima(sectores_df, config, usar_openmeteo=True):
    """
    Carga pronóstico horario para todos los sectores. Se filtran las horas
    anteriores al momento de ejecución para trabajar con un horizonte operacional
    desde la hora actual en adelante.
    """

    if usar_openmeteo:
        try:
            dfs = []

            for _, sector in sectores_df.iterrows():
                df_s = descargar_clima_openmeteo_forecast_sector(
                    sector,
                    config["FORECAST_DAYS"],
                    config["TIMEZONE"]
                )

                dfs.append(df_s)

            clima = pd.concat(dfs, ignore_index=True)
            clima = clima.sort_values(["datetime", "sector_id"]).reset_index(drop=True)

            # Se trabaja desde la hora actual local en adelante.
            ahora_local = pd.Timestamp.now(
                tz=config["TIMEZONE"]
            ).floor("h").tz_localize(None)

            clima = clima[clima["datetime"] >= ahora_local].copy()
            clima = clima.sort_values(["datetime", "sector_id"]).reset_index(drop=True)

            print("Datos climáticos cargados desde Open-Meteo Forecast API.")
            print("Inicio del horizonte climático:", clima["datetime"].min())
            print("Fin del horizonte climático:", clima["datetime"].max())

            return clima

        except Exception as e:
            print("No fue posible consultar Open-Meteo Forecast API.")
            print("Se usará una serie climática sintética reproducible.")
            print("Detalle:", e)

    return generar_clima_simulado(sectores_df, config, seed=config["SEED"])


def normalizar_clip(x, xmin, xmax):
    """Normaliza a [0,1] con recorte. Evita valores negativos o mayores a 1."""
    return np.clip((x - xmin) / (xmax - xmin), 0, 1)


def calcular_riesgo_climatico(clima_df):
    """
    Riesgo climático horario por sector.

    Criterio:
    - mayor temperatura aumenta riesgo;
    - menor humedad aumenta riesgo;
    - mayor viento aumenta riesgo;
    - precipitación reduce riesgo;
    - condiciones secas + viento generan un término de interacción.

    Los rangos son heurísticos:
    no buscan reemplazar índices oficiales, sino construir una señal
    operacional razonable para priorizar vigilancia preventiva.
    """
    df = clima_df.copy()

    df["temp_norm"] = normalizar_clip(df["temperatura"], 12, 34)
    df["humedad_baja_norm"] = normalizar_clip(100 - df["humedad"], 25, 80)
    df["viento_norm"] = normalizar_clip(df["viento"], 5, 45)
    df["precip_norm"] = normalizar_clip(df["precipitacion"], 0, 4)

    # Interacción: viento sobre vegetación/ambiente seco.
    df["seco_viento_norm"] = (
        df["humedad_baja_norm"]
        * (0.55 * df["viento_norm"] + 0.45 * df["temp_norm"])
    ).clip(0, 1)

    riesgo = (
        0.25 * df["temp_norm"]
        + 0.30 * df["humedad_baja_norm"]
        + 0.25 * df["viento_norm"]
        + 0.20 * df["seco_viento_norm"]
        - 0.30 * df["precip_norm"]
    )

    df["riesgo_climatico"] = riesgo.clip(0, 1)

    return df


def construir_clima_df(sectores_df, config, usar_openmeteo=True):
    clima_df = cargar_clima(sectores_df, config, usar_openmeteo=usar_openmeteo)

    # Verificación de horizonte disponible.
    horas_necesarias = config["K_SIM"] + config["HORIZON"]
    horas_disponibles = clima_df["datetime"].nunique()

    if horas_disponibles < horas_necesarias:
        print("La serie climática disponible no cubre todo el horizonte requerido.")
        print("Horas disponibles:", horas_disponibles, "| horas necesarias:", horas_necesarias)
        print("Se usará una serie climática sintética reproducible.")
        clima_df = generar_clima_simulado(sectores_df, config, seed=config["SEED"] + 99)

    clima_df = calcular_riesgo_climatico(clima_df)
    clima_df = clima_df.merge(
        sectores_df[["sector_id", "riesgo_hist", "vulnerabilidad", "nombre", "comuna"]],
        on="sector_id",
        how="left"
    )

    return clima_df
