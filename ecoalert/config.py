# Parámetros generales: horizonte, pesos de prioridad, restricciones climáticas, batería y rutas de datos.

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Licencia Gurobi (WLS) en la carpeta del proyecto
GUROBI_LIC_PATH = BASE_DIR / "gurobi.lic"
if not GUROBI_LIC_PATH.exists():
    raise FileNotFoundError(f"No se encontró la licencia Gurobi en {GUROBI_LIC_PATH}")
os.environ["GRB_LICENSE_FILE"] = str(GUROBI_LIC_PATH)

MAPA_RIESGO_PATH = BASE_DIR / "data" / "mapas.geojson"
CONAF_RIESGO_PATH = BASE_DIR / "data" / "riesgo_historico_conaf.csv"

# Carpeta de resultados local
OUTPUT_DIR_RESULTADOS = str(BASE_DIR / "resultados")

CONFIG = {
    # Pronóstico horario desde Open-Meteo Forecast API.
    # Se consulta al momento de ejecutar la simulación y se usa como
    # horizonte climático para la simulación operacional.
    "CLIMATE_MODE": "forecast_horizonte",
    "FORECAST_DAYS": 5,
    "TIMEZONE": "America/Santiago",

    # Cada k representa una hora operacional.
    "DELTA_MIN": 60,

    # Horizonte rodante: planifica 6 horas, ejecuta solo la primera.
    "HORIZON": 6,

    # 3 días completos hora a hora.
    "K_SIM": 72,

    # Estaciones de carga fijas.
    "N_HUBS": 4,

    # --------------------------------------------------------
    # Prioridad operacional
    # --------------------------------------------------------
    # El histórico CONAF entrega el riesgo territorial base. La decisión
    # hora a hora también considera clima, tiempo sin monitoreo y operación reciente.
    "PESO_RIESGO_HIST": 0.20,
    "PESO_RIESGO_CLIMA": 0.32,
    "PESO_AGE": 0.38,
    "PESO_VULNERABILIDAD": 0.10,
    "AGE_NORMALIZACION_MIN": 240.0,
    "UMBRAL_ABANDONO_MIN": 180.0,
    "BOOST_ABANDONO_MAX": 0.25,
    "VENTANA_RECIEN_MONITOREADO_MIN": 60.0,
    "PENALIZACION_RECIENTE": 0.10,
    "PRIORITY_THRESHOLD": 0.50,

    # --------------------------------------------------------
    # Deuda operacional
    # --------------------------------------------------------
    # 120 min: alerta temprana solo si el sector ya tiene prioridad alta.
    # 180 min: deuda operacional base.
    # 240 min: deuda crítica por abandono prolongado.
    "DEUDA_ALERTA_TEMPRANA_MIN": 120.0,
    "DEUDA_OPERACIONAL_MIN": 180.0,
    "DEUDA_CRITICA_MIN": 240.0,
    "DEUDA_PRIORIDAD_MIN": 0.60,
    "DEUDA_RELATIVA_SLA": 0.75,
    "GAMMA_DEUDA_NO_ATENDIDA": 1800.0,
    "BETA_DEUDA_MONITOR": 45.0,

    # --------------------------------------------------------
    # Pesos de la función objetivo del solver
    # --------------------------------------------------------
    "LAMBDA_TRAVEL": 0.040,
    "MU_SLACK": 260.0,
    "ETA_BATTERY_DEFICIT": 0.22,
    "BETA_MONITORING_REWARD": 30.0,
    "BETA_CRITICAL_REWARD": 16.0,
    "URGENT_PRIORITY_THRESHOLD": 0.60,
    "GAMMA_URGENT_MISS": 650.0,
    "BIG_M": 10000.0,

    # --------------------------------------------------------
    # Condiciones meteorológicas operacionales
    # --------------------------------------------------------
    "MAX_WIND_KMH": 42.0,
    "MAX_PRECIP_MM": 5.0,

    # --------------------------------------------------------
    # SLA técnico para auditoría del modelo
    # --------------------------------------------------------
    # Se usa en el panel y en la penalización interna, pero no en el hover
    # principal del mapa operativo.
    "SLA_BASE_MIN": 360.0,
    "SLA_REDUCCION_RIESGO_MIN": 180.0,
    "SLA_MIN_MIN": 120.0,

    # --------------------------------------------------------
    # Gestión energética
    # --------------------------------------------------------
    "USAR_ZONAS_OPERACIONALES_COBERTURA": True,
    "BATERIA_RESERVA_POST_MISION": 25.0,
    "BATERIA_UMBRAL_CARGA": 45.0,
    "BATERIA_FORZAR_CARGA": 30.0,
    "BATERIA_OBJETIVO_OPERACIONAL": 70.0,
    "BATERIA_MIN_SALIDA_NORMAL": 55.0,
    "BATERIA_MIN_SALIDA_URGENTE": 35.0,
    "MAX_VUELOS_CONSECUTIVOS": 2,
    "PENALIZACION_USO_RECIENTE_DRON": 7.0,
    "PENALIZACION_BATERIA_MEDIA": 10.0,
    "PENALIZACION_BATERIA_BAJA": 22.0,
    "CARGA_POR_HORA_DEFAULT": 38.0,

    # Reproducibilidad de componentes simuladas.
    "SEED": 11,

    # Exportación mínima.
    "EXPORTAR_RESULTADOS": True,
    "OUTPUT_DIR": OUTPUT_DIR_RESULTADOS,
}

CONFIG_MAPA = {
    "BBOX_OPERATIVO": (-71.72, -33.35, -71.17, -32.70),
    "N_SECTORES": 25,
    "RADIO_OPERATIVO_M": 3000,
    "CRS_METRICO": 32719,
    "PESO_MAPA": 0.70,
    "PESO_COMUNAL": 0.30,
    "VULNERABILIDAD_DEFAULT": 0.70,
    "USAR_GEOCODIFICACION": True,
    "SCORE_GRID_CODE": {2: 0.2, 3: 0.4, 4: 0.6, 5: 0.8, 6: 1.0},
}
