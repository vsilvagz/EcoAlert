# Punto de entrada de la app Streamlit: barra lateral, pestañas y orquestación de la simulación.

import contextlib
import io
import os
from pathlib import Path

import streamlit as st

from ecoalert.config import CONFIG
from ecoalert.clima import construir_clima_df
from ecoalert.simulacion import construir_escenario, simular_horizonte_rodante
from ecoalert.resultados import (
    SIMULACION_PATH,
    cargar_simulacion,
    consolidar_resultados,
    exportar_resultados,
    guardar_simulacion,
)
from ecoalert.graficos import (
    mapa_riesgo,
    visualizar_ciclo_mapa,
    visualizar_metricas_operativas,
    visualizar_panel_operativo,
    visualizar_recursos_vuelo,
)

st.set_page_config(page_title="EcoAlert", layout="wide")

ESTILO_PATH = Path(__file__).resolve().parent / "assets" / "estilo.css"
st.markdown(f"<style>{ESTILO_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

TABLAS = {
    "Acciones de drones": ("acciones_df", "acciones_drones.csv"),
    "Estado de drones por ciclo": ("drones_estado_df", "estado_drones_por_k.csv"),
    "Estado de sectores por ciclo": ("sectores_estado_df", "estado_sectores_por_k.csv"),
    "Eventos simulados": ("eventos_df", "eventos_simulados.csv"),
    "Métricas operativas": ("metricas_df", "metricas_operativas.csv"),
    "Sectores con riesgo CONAF": ("sectores_df", "sectores_con_riesgo_conaf.csv"),
    "Estaciones de carga": ("hubs_df", "estaciones_carga.csv"),
    "Flota simulada": ("drones_df", "drones_simulados.csv"),
    "Pronóstico climático usado": ("clima_df", "clima_pronostico_usado.csv"),
    "Riesgo histórico CONAF por comuna": ("conaf_riesgo_df", "riesgo_historico_conaf_por_comuna.csv"),
}


@st.cache_data(show_spinner="Construyendo escenario: riesgo CONAF, sectores y estaciones de carga...")
def obtener_escenario():
    return construir_escenario()


@st.cache_data(show_spinner="Obteniendo pronóstico climático...", ttl=3600)
def obtener_clima(_sectores_df, k_sim, usar_openmeteo):
    config = {**CONFIG, "K_SIM": k_sim}
    salida = io.StringIO()
    with contextlib.redirect_stdout(salida):
        clima_df = construir_clima_df(_sectores_df, config, usar_openmeteo=usar_openmeteo)
    sintetico = (not usar_openmeteo) or ("sintética" in salida.getvalue())
    return clima_df, sintetico


@st.cache_data(show_spinner="Dibujando mapa de riesgo...")
def obtener_html_mapa_riesgo(_capa_riesgo, _sectores_df):
    return mapa_riesgo(_capa_riesgo, _sectores_df).get_root().render()


def nueva_simulacion(res):
    st.session_state["res"] = res
    for clave in ("k_mapa", "k_panel"):
        st.session_state.pop(clave, None)


def mostrar_figura(fig):
    fig.update_layout(font=dict(family="Arial, Helvetica, sans-serif"))
    st.plotly_chart(fig, width="stretch")


def mover_k(clave, delta, k_max):
    st.session_state[clave] = min(max(st.session_state[clave] + delta, 0), k_max)


def selector_ciclo(clave, k_max, inicial=0):
    st.session_state.setdefault(clave, min(inicial, k_max))
    anterior, centro, siguiente = st.columns([1, 6, 1], vertical_alignment="bottom")
    anterior.button("← k anterior", key=f"{clave}_ant", on_click=mover_k, args=(clave, -1, k_max), width="stretch")
    siguiente.button("k siguiente →", key=f"{clave}_sig", on_click=mover_k, args=(clave, 1, k_max), width="stretch")
    centro.slider("Ciclo k", 0, k_max, key=clave)
    return st.session_state[clave]


with st.sidebar:
    st.header("Simulación")

    k_sim = st.slider("Ciclos (horas operacionales)", 6, 72, CONFIG["K_SIM"])
    fuente = st.radio("Fuente climática", ["Open-Meteo (pronóstico)", "Simulada (contingencia)"])

    st.subheader("Pesos de prioridad")
    peso_hist = st.slider("Riesgo histórico CONAF", 0.0, 1.0, CONFIG["PESO_RIESGO_HIST"], 0.01)
    peso_clima = st.slider("Riesgo climático", 0.0, 1.0, CONFIG["PESO_RIESGO_CLIMA"], 0.01)
    peso_age = st.slider("Tiempo sin monitoreo", 0.0, 1.0, CONFIG["PESO_AGE"], 0.01)
    peso_vuln = st.slider("Vulnerabilidad", 0.0, 1.0, CONFIG["PESO_VULNERABILIDAD"], 0.01)
    st.caption(f"Suma de pesos: {peso_hist + peso_clima + peso_age + peso_vuln:.2f}")

    ejecutar = st.button("Ejecutar simulación", type="primary", width="stretch")

    if os.path.exists(SIMULACION_PATH):
        if st.button("Cargar última simulación", width="stretch"):
            nueva_simulacion(cargar_simulacion())

if ejecutar:
    escenario = obtener_escenario()
    clima_df, clima_sintetico = obtener_clima(
        escenario["sectores_df"], k_sim, fuente.startswith("Open-Meteo")
    )

    config = {
        **CONFIG,
        "K_SIM": k_sim,
        "PESO_RIESGO_HIST": peso_hist,
        "PESO_RIESGO_CLIMA": peso_clima,
        "PESO_AGE": peso_age,
        "PESO_VULNERABILIDAD": peso_vuln,
    }

    barra = st.progress(0.0, text="Resolviendo ciclo 0...")

    def progreso(i, n):
        barra.progress(i / n, text=f"Resolviendo ciclo {i} de {n}")

    snapshots = simular_horizonte_rodante(
        config,
        escenario["drones_df"],
        escenario["sectores_df"],
        escenario["hubs_df"],
        escenario["nodes_df"],
        escenario["dist_df"],
        clima_df,
        progreso=progreso,
    )
    barra.empty()

    res = consolidar_resultados(snapshots, escenario, clima_df, config)
    res["clima_sintetico"] = clima_sintetico
    guardar_simulacion(res)
    nueva_simulacion(res)

res = st.session_state.get("res")

if res is None:
    st.info("Configura los parámetros en la barra lateral y presiona **Ejecutar simulación**.")
    st.stop()

snapshots = res["snapshots"]
metricas_df = res["metricas_df"]
k_max = len(snapshots) - 1

clima_txt = "serie climática sintética" if res.get("clima_sintetico") else "pronóstico Open-Meteo"
st.caption(
    f"{len(snapshots)} ciclos · {snapshots[0]['datetime']} → {snapshots[-1]['datetime']} · Clima: {clima_txt}"
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Cobertura prioritaria (prom.)", f"{metricas_df['cobertura_prioritaria'].mean():.0%}")
c2.metric("Vuelos de monitoreo", int(metricas_df["vuelos"].sum()))
c3.metric("Cargas", int(metricas_df["cargas"].sum()))
c4.metric("Batería mínima", f"{metricas_df['bateria_minima'].min():.0f}%")
c5.metric("Indicios de incendio", int(metricas_df["eventos"].sum()))

tab_mapa, tab_panel, tab_metricas, tab_riesgo, tab_datos = st.tabs(
    ["Mapa operativo", "Panel técnico", "Métricas", "Mapa de riesgo", "Datos"]
)

with tab_mapa:
    k = selector_ciclo("k_mapa", k_max)
    mostrar_traces_drones = st.checkbox("Mostrar rutas/acciones de drones", value=True)
    st.caption(f"Hora operacional: {snapshots[k]['datetime']}")
    mostrar_figura(
        visualizar_ciclo_mapa(
            snapshots,
            k=k,
            mostrar_traces_drones=mostrar_traces_drones,
            mostrar_flechas_rutas=True,
        )
    )

with tab_panel:
    st.write(
        "Este panel se mantiene separado del mapa. "
        "Permite auditar si la prioridad proviene del histórico CONAF, del clima o del tiempo sin monitoreo."
    )
    k_panel = selector_ciclo("k_panel", k_max, inicial=24)
    st.caption(f"Hora operacional: {snapshots[k_panel]['datetime']}")
    mostrar_figura(visualizar_panel_operativo(snapshots, k=k_panel))

with tab_metricas:
    mostrar_figura(visualizar_metricas_operativas(metricas_df))
    mostrar_figura(visualizar_recursos_vuelo(metricas_df))

with tab_riesgo:
    if st.toggle("Mostrar mapa de riesgo (amenaza CONAF/SENAPRED)"):
        st.iframe(
            obtener_html_mapa_riesgo(obtener_escenario()["capa_riesgo"], res["sectores_df"]),
            height=650,
        )

with tab_datos:
    nombre = st.selectbox("Tabla", list(TABLAS))
    clave, archivo = TABLAS[nombre]
    df = res[clave]
    st.dataframe(df, width="stretch")
    st.download_button("Descargar CSV", df.to_csv(index=False).encode("utf-8"), file_name=archivo, mime="text/csv")

    if st.button("Exportar todos los CSV a la carpeta de resultados"):
        st.success(f"Exportación completada en: {exportar_resultados(res)}")
