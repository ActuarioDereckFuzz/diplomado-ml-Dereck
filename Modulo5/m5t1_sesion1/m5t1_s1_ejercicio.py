"""
EJERCICIO — Módulo 5 · Tema 1 · Sesión 1
========================================
Diplomado Machine Learning en Seguros · FC UNAM

Extiende el tablero mínimo. Este archivo YA CORRE, pero está INCOMPLETO: busca los
comentarios  # TODO  y complétalos. La solución está en m5t1_s1_soluciones.py.

Correr:  streamlit run m5t1_s1_ejercicio.py   (NO con 'python')

Tareas
------
  1. Validar la frecuencia de la cartera contra el número de M4 (~0.1392).
  2. Agregar un filtro de antigüedad del vehículo en la barra lateral.
  3. Agregar una tarjeta de KPI con la prima pura.
  4. Graficar la severidad media por sexo.
  ★ Reto: que el gráfico cambie de factor con un st.selectbox.

Al terminar: commit y push a tu repositorio de GitHub.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Ejercicio S1", layout="wide")
BANDAS_EDAD  = [17, 30, 35, 45, 50, 55, 60, 95]
BANDAS_ANTIG = [-1, 1, 2, 3, 4, 5, 10, 15, 50]   # cortes de M4


def ruta_datos():
    for r in ["../../Modulo_4/m4t2_sesion1/datos/datos.parquet"]:
        if os.path.exists(r):
            return r
    raise FileNotFoundError("No encuentro datos.parquet.")


@st.cache_data
def cargar_datos():
    df = pd.read_parquet(ruta_datos())
    df["edad_cat"] = pd.cut(df["edad_conductor"], bins=BANDAS_EDAD)
    # TODO 2a: crea la columna df["antig_cat"] con pd.cut sobre "antiguedad_vehiculo"
    #          usando BANDAS_ANTIG (mira cómo se hizo arriba con edad_cat).
    return df


# --- KPIs (ya dados: son los mismos del notebook) ---
def kpi_frecuencia(df):
    return df["num_siniestros"].sum() / df["exposicion"].sum()

def kpi_severidad_media(df):
    con = df[df["num_siniestros"] > 0]          # severidad es NaN cuando no hubo siniestro
    return np.average(con["severidad"], weights=con["num_siniestros"]) if len(con) else np.nan

def kpi_prima_pura(df):
    return df["monto_total"].sum() / df["exposicion"].sum()

def kpis_por(df, var):
    filas = []
    for niv, g in df.groupby(var, observed=True):
        filas.append({var: str(niv), "sev": kpi_severidad_media(g), "freq": kpi_frecuencia(g)})
    return pd.DataFrame(filas)


datos = cargar_datos()
st.title("📊 Ejercicio — extiende el tablero")

# ── TODO 1 · Validación ───────────────────────────────────────────────────────
# Muestra un st.caption que compare kpi_frecuencia(datos) con 0.1392 (tolerancia 0.001)
# y diga si coincide.
# TODO 1: escribe aquí tu st.caption(...)


# ── Filtros ───────────────────────────────────────────────────────────────────
st.sidebar.header("Filtros")
cob = st.sidebar.multiselect("Cobertura", sorted(datos["cobertura"].unique()),
                             default=sorted(datos["cobertura"].unique()))
# TODO 2b: agrega un st.sidebar.multiselect para la antigüedad usando las categorías
#          de datos["antig_cat"] (recuerda crear la columna en TODO 2a).

df = datos[datos["cobertura"].isin(cob)]
# TODO 2c: aplica también el filtro de antigüedad al df.

if len(df) == 0:
    st.warning("Ningún registro con esos filtros."); st.stop()

# ── Tarjetas ──────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric("Frecuencia", f"{kpi_frecuencia(df):.4f}")
c2.metric("Severidad media", f"${kpi_severidad_media(df):,.0f}")
# TODO 3: usa c3 para mostrar la PRIMA PURA con kpi_prima_pura(df), formateada como $.


# ── Gráfico ───────────────────────────────────────────────────────────────────
st.subheader("Severidad media por sexo")
# TODO 4: construye la tabla con kpis_por(df, "sexo") y dibújala con ax.bar(...).
#         Pista:
#   tab = kpis_por(df, "sexo")
#   fig, ax = plt.subplots(figsize=(6, 3.5))
#   ax.bar(tab["sexo"], tab["sev"], color="#4C72B0")
#   st.pyplot(fig)
#
# ★ Reto: en vez de fijar "sexo", deja que el usuario elija el factor con
#         st.selectbox(["sexo", "cobertura", "uso"]) y grafica ese.
