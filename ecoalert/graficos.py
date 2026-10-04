# Figuras Plotly: mapa operativo, panel técnico-operativo y gráficos de métricas.

import numpy as np
import geopandas as gpd
import folium
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def visualizar_metricas_operativas(metricas_df):
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Cobertura de sectores prioritarios",
            "Sectores críticos y sobre SLA",
            "Batería promedio y mínima",
            "Actividad de la flota"
        ),
        vertical_spacing=0.16,
        horizontal_spacing=0.10
    )

    fig.add_trace(
        go.Scatter(
            x=metricas_df["k"],
            y=metricas_df["cobertura_prioritaria"],
            mode="lines+markers",
            name="Cobertura prioritaria"
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=metricas_df["k"],
            y=metricas_df["pendientes_criticos"],
            mode="lines+markers",
            name="Pendientes críticos"
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=metricas_df["k"],
            y=metricas_df["sectores_sobre_SLA"],
            mode="lines+markers",
            name="Sobre SLA"
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=metricas_df["k"],
            y=metricas_df["bateria_promedio"],
            mode="lines+markers",
            name="Batería promedio"
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=metricas_df["k"],
            y=metricas_df["bateria_minima"],
            mode="lines+markers",
            name="Batería mínima"
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Bar(
            x=metricas_df["k"],
            y=metricas_df["vuelos"],
            name="Vuelos/monitoreos"
        ),
        row=2, col=2
    )
    fig.add_trace(
        go.Bar(
            x=metricas_df["k"],
            y=metricas_df["cargas"],
            name="Cargas"
        ),
        row=2, col=2
    )
    fig.add_trace(
        go.Bar(
            x=metricas_df["k"],
            y=metricas_df["esperas"],
            name="Esperas"
        ),
        row=2, col=2
    )

    fig.update_layout(
        height=860,
        width=1250,
        title="EcoAlert — Métricas simuladas",
        template="plotly_white",
        barmode="stack",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.16,
            xanchor="left",
            x=0,
            font=dict(size=11),
            entrywidth=170,
            entrywidthmode="pixels"
        ),
        margin=dict(l=60, r=35, t=85, b=155)
    )
    fig.update_yaxes(title_text="Fracción", range=[0, 1.05], row=1, col=1)
    fig.update_yaxes(title_text="Sectores", row=1, col=2)
    fig.update_yaxes(title_text="Batería (%)", range=[0, 105], row=2, col=1)
    fig.update_yaxes(title_text="Cantidad", row=2, col=2)
    return fig


def visualizar_recursos_vuelo(metricas_df):
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Distancia recorrida por ciclo", "Consumo energético estimado por ciclo"),
        horizontal_spacing=0.12
    )
    fig.add_trace(
        go.Scatter(
            x=metricas_df["k"],
            y=metricas_df["distancia_total_km"],
            mode="lines+markers",
            name="Distancia total"
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=metricas_df["k"],
            y=metricas_df["consumo_estimado_pct"],
            mode="lines+markers",
            name="Consumo estimado"
        ),
        row=1, col=2
    )
    fig.update_layout(
        height=540,
        width=1250,
        title="EcoAlert — Uso simulado de recursos de vuelo",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="left",
            x=0,
            font=dict(size=11),
            entrywidth=190,
            entrywidthmode="pixels"
        ),
        margin=dict(l=60, r=35, t=85, b=125)
    )
    fig.update_yaxes(title_text="km", row=1, col=1)
    fig.update_yaxes(title_text="% batería estimado", row=1, col=2)
    return fig


# Nota: se usa go.Scattermap / layout.map (MapLibre) en lugar de go.Scattermapbox /
# layout.mapbox, que fueron eliminados en plotly >= 7.
# Scattermap está disponible desde plotly 5.24.

def obtener_coords_node(nodes_df):
    return {
        r.node_id: {
            "lat": float(r.lat),
            "lon": float(r.lon),
            "nombre": r.nombre,
            "type": r.node_type
        }
        for _, r in nodes_df.iterrows()
    }


def add_arrowhead_mapbox(
    fig,
    lat_start,
    lon_start,
    lat_end,
    lon_end,
    color="#2A9D8F",
    width=4,
    arrow_frac=0.12,
    arrow_angle_deg=28,
    arrow_min=0.015,
    arrow_max=0.045,
):
    """
    Agrega una punta de flecha simulada al final de un tramo en Scattermapbox.

    Plotly no incorpora flechas nativas para rutas en mapas Mapbox, por lo que
    se dibujan dos segmentos cortos al final del tramo. La punta queda integrada
    al visualizador.
    """
    lat_start = float(lat_start)
    lon_start = float(lon_start)
    lat_end = float(lat_end)
    lon_end = float(lon_end)

    # Aproximación local equirectangular para calcular dirección sobre el mapa.
    mean_lat_rad = np.deg2rad((lat_start + lat_end) / 2)
    cos_lat = max(np.cos(mean_lat_rad), 1e-6)
    dx = (lon_end - lon_start) * cos_lat
    dy = lat_end - lat_start
    segment_len = float(np.hypot(dx, dy))

    # Si el tramo es prácticamente nulo, no se dibuja flecha.
    if segment_len < 1e-9:
        return fig

    theta = np.arctan2(dy, dx)
    alpha = np.deg2rad(arrow_angle_deg)
    arrow_len = float(np.clip(segment_len * arrow_frac, arrow_min, arrow_max))

    # Brazos de la flecha, apuntando hacia atrás desde el destino del tramo.
    arm_angles = [theta + np.pi - alpha, theta + np.pi + alpha]
    lats = []
    lons = []

    for ang in arm_angles:
        arm_dx = arrow_len * np.cos(ang)
        arm_dy = arrow_len * np.sin(ang)
        arm_lat = lat_end + arm_dy
        arm_lon = lon_end + arm_dx / max(np.cos(np.deg2rad(lat_end)), 1e-6)

        lats.extend([lat_end, arm_lat, None])
        lons.extend([lon_end, arm_lon, None])

    fig.add_trace(go.Scattermap(
        lat=lats,
        lon=lons,
        mode="lines",
        line=dict(width=width, color=color),
        hoverinfo="skip",
        showlegend=False,
        name="Dirección de ruta"
    ))

    return fig


def visualizar_ciclo_mapa(snapshots, k=0, mostrar_traces_drones=True, mostrar_flechas_rutas=True):
    """
    Visualiza el mapa operativo limpio para un ciclo k.

    Esta función está pensada como vista principal de la app:
    - sectores coloreados por prioridad;
    - estaciones de carga fijas;
    - rutas de monitoreo visibles u ocultables desde el selector;
    - puntos de carga y espera;
    - hover operativo con clima, riesgo, tiempo sin monitoreo y drones factibles.
      Los detalles internos del modelo quedan en el panel técnico.

    Parámetros clave:
    - mostrar_traces_drones=True: muestra rutas/acciones de drones.
    - mostrar_traces_drones=False: oculta todos los traces de drones para inspeccionar sectores/hubs sin superposición.
    - mostrar_flechas_rutas=True: agrega una punta de flecha en el tramo hub/posición inicial → sector para indicar dirección de salida.
    """
    snap = snapshots[k]
    data = snap["data"]
    result = snap["result"]

    nodes_df = data["nodes_df"]
    sectores_df = data["sectores_df"]
    hubs_df = data["hubs_df"]
    coords = obtener_coords_node(nodes_df)

    fig = go.Figure()

    # --------------------------------------------------------
    # 1) Sectores de monitoreo
    # --------------------------------------------------------
    prioridades = []
    hover_text = []

    for _, s in sectores_df.iterrows():
        g = s["sector_id"]
        prioridad_g = data["w"][(g, 0)]
        prioridades.append(prioridad_g)

        clima_g = data["clima_node"][(g, 0)]
        comps = data.get("priority_components", {}).get((g, 0), {})
        tag_prioritario = "Sí" if prioridad_g >= data.get("priority_threshold", 0.50) else "No"
        monitoreado = "Sí" if result["y"][(g, 0)] > 0.5 else "No"
        urgente = "Sí" if data.get("urgent", {}).get((g, 0), 0) == 1 else "No"
        drones_factibles = data.get("feasible_drones", {}).get((g, 0), [])
        drones_factibles_txt = ", ".join(drones_factibles) if drones_factibles else "Sin dron factible"

        estado_operativo = "Monitoreado" if monitoreado == "Sí" else ("Urgente" if urgente == "Sí" else ("Prioritario" if tag_prioritario == "Sí" else "Normal"))
        hover_text.append(
            f"<b>{g} - {s['nombre']}</b><br>"
            f"Comuna: {s['comuna']}<br>"
            f"Estado: {estado_operativo}<br>"
            f"Prioridad operacional: {prioridad_g:.2f}<br>"
            f"Riesgo histórico CONAF: {s['riesgo_hist']:.2f}<br>"
            f"Riesgo climático: {clima_g['riesgo_climatico']:.2f}<br>"
            f"Tiempo sin monitoreo: {data['age0'][g]:.0f} min<br>"
            f"Drones factibles: {drones_factibles_txt}<br>"
            f"Temp.: {clima_g['temperatura']:.1f} °C<br>"
            f"Humedad: {clima_g['humedad']:.1f}%<br>"
            f"Viento: {clima_g['viento']:.1f} km/h<br>"
            f"Precip.: {clima_g['precipitacion']:.2f} mm<br>"
            f"Presión: {clima_g.get('presion', np.nan):.0f} hPa"
        )

    fig.add_trace(go.Scattermap(
        lat=sectores_df["lat"],
        lon=sectores_df["lon"],
        mode="markers+text",
        text=sectores_df["sector_id"],
        textposition="top center",
        marker=dict(
            size=17,
            color=prioridades,
            colorscale="YlOrRd",
            cmin=0,
            cmax=1,
            showscale=True,
            colorbar=dict(title="Prioridad")
        ),
        hovertext=hover_text,
        hoverinfo="text",
        name="Sectores"
    ))

    # --------------------------------------------------------
    # 2) Estaciones de carga
    # --------------------------------------------------------
    fig.add_trace(go.Scattermap(
        lat=hubs_df["lat"],
        lon=hubs_df["lon"],
        mode="markers+text",
        text=hubs_df["hub_id"],
        textposition="bottom center",
        marker=dict(size=19, color="blue"),
        hovertext=[
            f"<b>{r.hub_id}</b><br>{r.nombre}<br>Capacidad: {r.capacidad}"
            for _, r in hubs_df.iterrows()
        ],
        hoverinfo="text",
        name="Estaciones de carga"
    ))

    # --------------------------------------------------------
    # 3) Rutas y estados de drones, opcionales
    # --------------------------------------------------------
    colores_dron = {
        "D1": "#00A878",
        "D2": "#9B5DE5",
        "D3": "#FF9F1C",
        "D4": "#00B4D8",
        "D5": "#EF476F",
        "D6": "#7AC943"
    }

    if mostrar_traces_drones:
        for a in result["actions"]:
            d = a["dron"]
            color = colores_dron.get(d, "#2A9D8F")

            if a["tipo"] == "monitoreo":
                h = a["desde"]
                g = a["sector"]
                r = a.get("retorno", h)

                lat_path = [coords[h]["lat"], coords[g]["lat"], coords[r]["lat"]]
                lon_path = [coords[h]["lon"], coords[g]["lon"], coords[r]["lon"]]

                hover = (
                    f"<b>{d}</b><br>"
                    f"Acción: monitoreo<br>"
                    f"Salida: {h}<br>"
                    f"Sector: {g}<br>"
                    f"Retorno: {r}<br>"
                    f"Prioridad sector: {data['w'][(g, 0)]:.2f}<br>"
                    f"Batería post: {snap['estado_post']['bateria'][d]:.1f}%"
                )

                fig.add_trace(go.Scattermap(
                    lat=lat_path,
                    lon=lon_path,
                    mode="lines+markers",
                    line=dict(width=4, color=color),
                    marker=dict(size=9, color=color),
                    name=f"{d}: {h}->{g}->{r}",
                    hovertext=hover,
                    hoverinfo="text"
                ))

                # Flecha de dirección integrada al mapa.
                # Se muestra solo en el tramo hub/posición inicial → sector,
                # ya que el retorno al hub queda implícito en la ruta.
                if mostrar_flechas_rutas:
                    add_arrowhead_mapbox(
                        fig,
                        coords[h]["lat"], coords[h]["lon"],
                        coords[g]["lat"], coords[g]["lon"],
                        color=color,
                        width=4
                    )

            elif a["tipo"] == "carga":
                h = a["carga_en"]

                hover = (
                    f"<b>{d}</b><br>"
                    f"Acción: carga<br>"
                    f"Estación: {h}<br>"
                    f"Batería post: {snap['estado_post']['bateria'][d]:.1f}%"
                )

                fig.add_trace(go.Scattermap(
                    lat=[coords[h]["lat"]],
                    lon=[coords[h]["lon"]],
                    mode="markers+text",
                    text=[f"{d} carga"],
                    textposition="top center",
                    marker=dict(size=15, color="purple"),
                    name=f"{d}: carga",
                    hovertext=hover,
                    hoverinfo="text"
                ))

            else:
                h = a["desde"]

                hover = (
                    f"<b>{d}</b><br>"
                    f"Acción: espera<br>"
                    f"Ubicación: {h}<br>"
                    f"Batería post: {snap['estado_post']['bateria'][d]:.1f}%"
                )

                fig.add_trace(go.Scattermap(
                    lat=[coords[h]["lat"]],
                    lon=[coords[h]["lon"]],
                    mode="markers+text",
                    text=[f"{d} espera"],
                    textposition="top center",
                    marker=dict(size=12, color="gray"),
                    name=f"{d}: espera",
                    hovertext=hover,
                    hoverinfo="text"
                ))

    # --------------------------------------------------------
    # 4) Layout
    # --------------------------------------------------------
    fig.update_layout(
        title=f"EcoAlert — Plataforma operativa | k={k} | {snap['datetime']}",
        map=dict(
            style="open-street-map",
            center=dict(
                lat=float(nodes_df["lat"].mean()),
                lon=float(nodes_df["lon"].mean())
            ),
            zoom=8.5
        ),
        height=720,
        width=1150,
        legend=dict(
            orientation="h",
            y=-0.12,
            x=0.0
        ),
        margin=dict(l=10, r=10, t=60, b=10)
    )

    return fig


def visualizar_panel_operativo(snapshots, k=0):
    snap = snapshots[k]
    data = snap["data"]
    G = data["G"]

    prioridades = [data["w"][(g, 0)] for g in G]
    age_pre = [data["age0"][g] for g in G]
    sla_vals = [data["L"][g] for g in G]
    riesgo_hist = [data["priority_components"][(g, 0)]["riesgo_hist"] for g in G]
    riesgo_clima = [data["priority_components"][(g, 0)]["riesgo_climatico"] for g in G]
    comp_hist = [data["priority_components"][(g, 0)]["comp_hist"] for g in G]
    comp_clima = [data["priority_components"][(g, 0)]["comp_clima"] for g in G]
    comp_age = [data["priority_components"][(g, 0)]["comp_age"] for g in G]
    boost = [data["priority_components"][(g, 0)]["abandono_boost"] for g in G]
    penal = [data["priority_components"][(g, 0)]["penalizacion_reciente"] for g in G]
    urgente = [data.get("urgent", {}).get((g, 0), 0) for g in G]
    drones_factibles = [data.get("coverage_count", {}).get((g, 0), 0) for g in G]
    monitoreado = [1 if snap["result"]["y"][(g, 0)] > 0.5 else 0 for g in G]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Prioridad operacional por sector",
            "Tiempo sin monitoreo vs SLA",
            "Riesgo histórico CONAF vs riesgo climático",
            "Componentes dinámicos de prioridad"
        )
    )

    fig.add_trace(
        go.Bar(x=G, y=prioridades, text=[f"{v:.2f}" for v in prioridades], textposition="auto", name="Prioridad"),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=G, y=monitoreado, mode="markers", marker=dict(size=12, symbol="star"), name="Monitoreado k"),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(x=G, y=age_pre, text=[f"{v:.0f}" for v in age_pre], textposition="auto", name="Age pre"),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=G, y=sla_vals, mode="lines+markers", name="SLA", line=dict(width=3)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=G, y=[u*max(sla_vals)*1.05 for u in urgente], mode="markers", marker=dict(size=10, symbol="diamond"), name="Urgente"),
        row=1, col=2
    )

    fig.add_trace(
        go.Bar(x=G, y=riesgo_hist, name="Riesgo histórico"),
        row=2, col=1
    )
    fig.add_trace(
        go.Bar(x=G, y=riesgo_clima, name="Riesgo climático"),
        row=2, col=1
    )

    fig.add_trace(
        go.Bar(x=G, y=comp_hist, name="Comp. histórico"),
        row=2, col=2
    )
    fig.add_trace(
        go.Bar(x=G, y=comp_clima, name="Comp. clima"),
        row=2, col=2
    )
    fig.add_trace(
        go.Bar(x=G, y=comp_age, name="Comp. age"),
        row=2, col=2
    )
    fig.add_trace(
        go.Bar(x=G, y=boost, name="Boost abandono"),
        row=2, col=2
    )
    fig.add_trace(
        go.Bar(x=G, y=[-p for p in penal], name="Penalización reciente"),
        row=2, col=2
    )

    fig.update_layout(
        height=780,
        width=1250,
        title=f"EcoAlert — Panel técnico-operativo separado | k={k} | {snap['datetime']}",
        template="plotly_white",
        barmode="group",
        legend=dict(orientation="h", y=-0.15, x=0)
    )
    fig.update_yaxes(title_text="Prioridad", row=1, col=1)
    fig.update_yaxes(title_text="Minutos", row=1, col=2)
    fig.update_yaxes(title_text="Riesgo normalizado", row=2, col=1)
    fig.update_yaxes(title_text="Contribución a prioridad", row=2, col=2)

    return fig


def mapa_riesgo(capa_riesgo, sectores_df):
    m = capa_riesgo.to_crs(4326).explore(
        "Peligrosidad", categorical=True, categories=["Baja", "Media", "Alta"],
        cmap=["green", "yellow", "red"], legend=True, name="Amenaza (CONAF/SENAPRED)",
        style_kwds={"fillOpacity": 0.35, "weight": 0.3},
    )
    gpd.GeoDataFrame(
        sectores_df, geometry=gpd.points_from_xy(sectores_df["lon"], sectores_df["lat"]), crs=4326,
    ).explore(
        m=m, color="black", marker_kwds={"radius": 7}, name="Sectores desde mapa (M)",
        tooltip=["sector_id", "nombre", "riesgo_mapa", "riesgo_hist"],
    )

    folium.LayerControl().add_to(m)

    return m
