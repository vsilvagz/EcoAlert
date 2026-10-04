# EcoAlert

Simula el monitoreo preventivo de incendios forestales con una flota de drones en la Región de Valparaíso.

La simulación combina riesgo histórico CONAF, pronóstico climático horario, estaciones de carga fijas y restricciones de batería. En cada ciclo horario un modelo MILP (Gurobi) decide si cada dron monitorea un sector, carga o espera, con horizonte rodante de 6 horas.

## Estructura

```text
app.py              Interfaz Streamlit
ecoalert/           Lógica del modelo, independiente de la interfaz
  config.py         Parámetros y rutas
  riesgo.py         Riesgo histórico CONAF y sectores desde el mapa de riesgo
  flota.py          Estaciones de carga y drones
  clima.py          Pronóstico horario y riesgo climático
  red.py            Nodos, distancias y estado inicial
  prioridad.py      Prioridad operacional y parámetros por ciclo
  solver.py         Modelo MILP
  simulacion.py     Horizonte rodante y eventos
  resultados.py     Tablas, métricas y exportación
  graficos.py       Figuras del mapa, panel técnico y métricas
data/               Datos de entrada
assets/             Recursos de la interfaz
.streamlit/         Tema y opciones de Streamlit
```

## Instalación

Requiere Python 3.10 o superior.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Requiere una licencia de Gurobi: el archivo `gurobi.lic` debe estar en la raíz del proyecto (junto a `app.py`).

## Ejecución

```powershell
streamlit run app.py
```
