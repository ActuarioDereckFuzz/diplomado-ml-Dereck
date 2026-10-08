"""
app_streamlit.py — Tablero mínimo · Módulo 5 · Tema 1 · Sesión 1
================================================================
Diplomado Machine Learning en Seguros · FC UNAM

Este es el "tablero mínimo" que construimos EN VIVO en la Sesión 1. Presenta los
KPIs de la cartera de M4 que definimos y validamos en `m5t1_s1_notebook.ipynb`.

    La app NO recalcula lógica nueva: usa las MISMAS funciones de KPI del notebook.
    El notebook define; la app presenta.

Cómo se corre (esto NO va en un notebook):
    conda activate diplomado
    streamlit run app_streamlit.py

Ideas de Streamlit que se enseñan aquí:
  · Streamlit corre un script de arriba a abajo y lo RE-EJECUTA completo en cada
    interacción (cada vez que mueves un filtro). Por eso cacheamos la carga de datos.
  · @st.cache_data guarda en memoria el resultado de una función cara (leer el archivo)
    para no releerlo en cada rerun.
  · Los widgets (st.sidebar.multiselect, st.slider) devuelven su valor actual; con ese
    valor filtramos el DataFrame y todo lo de abajo se recalcula solo.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# 0 · Configuración de la página
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Tablero de cartera — Diplomado ML Seguros",
                   page_icon="📊", layout="wide")

# Bandas de M4 (mismos cortes que en los GLM del Módulo 4)
BANDAS_EDAD = [17, 30, 35, 45, 50, 55, 60, 95]

# Abiqca tu ruta de trabajo a la carpeta del módulo 5, para que el notebook y la app corran sin errores.
# cd "C:\Users\Eric_Daniel\Documents\Ciencias Cursos\Diplomado\diplomado-ml-seguros\diplomado-ml-seguros\Modulo_5\m5t1_sesion1"
# cd "diplomado-ml-seguros\diplomado-ml-seguros\Modulo_5\m5t1_sesion3"
#se corre com:
# streamlit run app_streamlit.py



# ─────────────────────────────────────────────────────────────────────────────
# 1 · Carga de datos (cacheada) — la fuente de verdad es la cartera de M4
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def cargar_datos():
    """Lee la cartera. Acepta parquet (ligero) o pickle. Cacheada: se lee una sola vez."""
    candidatos = [
        "datos/datos.parquet", "datos/datos.pkl",
        "../../Modulo_4/m4t2_sesion1/datos/datos.parquet",
        "../../Modulo_4/m4t2_sesion1/datos/datos.pkl",
    ]
    ruta = next((r for r in candidatos if os.path.exists(r)), None)
    if ruta is None:
        st.error("No encuentro datos.parquet ni datos.pkl. Copia la cartera de M4 a Modulo_5/datos/.")
        st.stop()
    df = pd.read_parquet(ruta) if ruta.endswith(".parquet") else pd.read_pickle(ruta)
    df["edad_cat"] = pd.cut(df["edad_conductor"], bins=BANDAS_EDAD)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2 · KPIs — LAS MISMAS funciones del notebook (aquí solo se importan/reusan)
# ─────────────────────────────────────────────────────────────────────────────
def kpi_exposicion(df):      return df["exposicion"].sum()
def kpi_num_siniestros(df):  return int(df["num_siniestros"].sum())
def kpi_frecuencia(df):      return df["num_siniestros"].sum() / df["exposicion"].sum()

def kpi_severidad_media(df):
    """Ponderada por nº de siniestros, solo sobre pólizas con N>0 (severidad es NaN si N=0)."""
    con = df[df["num_siniestros"] > 0]
    if con["num_siniestros"].sum() == 0:
        return np.nan
    return np.average(con["severidad"], weights=con["num_siniestros"])

def kpi_prima_pura(df):      return df["monto_total"].sum() / df["exposicion"].sum()


def kpis_por(df, variable):
    """Tabla de KPIs por nivel de una variable, ordenada por exposición."""
    filas = []
    for nivel, g in df.groupby(variable, observed=True):
        filas.append({
            variable: str(nivel),
            "Exposición": kpi_exposicion(g),
            "Nº sin.": kpi_num_siniestros(g),
            "Frecuencia": kpi_frecuencia(g),
            "Severidad": kpi_severidad_media(g),
            "Prima pura": kpi_prima_pura(g),
        })
    return pd.DataFrame(filas).sort_values("Exposición", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3 · Cuerpo de la app
# ─────────────────────────────────────────────────────────────────────────────
datos = cargar_datos()

st.title("📊 Tablero de cartera — auto")
st.caption("Sesión 1 · tablero mínimo sobre la cartera del Módulo 4. Mueve los filtros de la izquierda.")

# --- Barra lateral: filtros ---------------------------------------------------
st.sidebar.header("Filtros")

# Filtros categóricos: se construyen a partir de los valores reales (no hardcode)
with st.sidebar.form("filtros"):
    cob = st.multiselect("Cobertura", options=sorted(datos["cobertura"].unique()), 
                         default=sorted(datos["cobertura"].unique()),
                         help = "Tipo de cobertura de la póliza")
    sexo = st.multiselect("Sexo", options=sorted(datos["sexo"].unique()), 
                          default=sorted(datos["sexo"].unique()),
                          help = "Sexo del contratante de la póliza")
    uso = st.multiselect("Uso", options=sorted(datos["uso"].unique()), 
                         default=sorted(datos["uso"].unique()),
                         help = "Uso del vehículo asegurado")
    comb = st.multiselect("Combustible", options=sorted(datos["combustible"].unique()), 
                          default=sorted(datos["combustible"].unique()),
                          help = "Tipo de combustible del vehículo asegurado")
    emin, emax = int(datos["edad_conductor"].min()), int(datos["edad_conductor"].max())
    edad = st.slider("Edad", min_value=emin, max_value=emax, value=(emin, emax),
                     help = "Rango de edad del conductor de la póliza")
    st.form_submit_button("Aplicar filtros")
    
df = datos[datos["cobertura"].isin(cob) & datos["sexo"].isin(sexo) & datos["uso"].isin(uso) & datos["combustible"].isin(comb)]
df = df[df["edad_conductor"].between(*edad)]

if len(df) == 0:
    st.warning("Ningún registro con esos filtros. Amplía la selección.")
    st.stop()
# --- Aplicar filtros (esto se re-ejecuta en cada interacción) -----------------

tab_cartera, tab_detalle, tab_cotiza = st.tabs(["Cartera", "Detalle", "Cotizador"])

with tab_cartera:
    df = df.copy()
    st.sidebar.metric("Pólizas seleccionadas", f"{len(df):,}")

    # --- Tarjetas de KPI (st.metric) ----------------------------------------------
    st.subheader("Indicadores de la selección")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Exposición", f"{kpi_exposicion(df):,.0f}")
    c2.metric("Nº siniestros", f"{kpi_num_siniestros(df):,}")
    c3.metric("Frecuencia", f"{kpi_frecuencia(df):.4f}",
              delta = f"{kpi_frecuencia(datos):.4f} respecto a la cartera completa")
    c4.metric("Severidad media", f"${kpi_severidad_media(df):,.0f}",
              delta = f"${kpi_severidad_media(df) - kpi_severidad_media(datos):,.0f}")
    c5.metric("Prima pura", f"${kpi_prima_pura(df):,.2f}")

    st.divider()

    # --- Dos columnas: gráficos ---------------------------------------------------
    g1, g2 = st.columns(2)

    with g1:
        with st.container(border = True):
            st.markdown("**Exposición por banda de edad**")
            tab_edad = kpis_por(df, "edad_cat")
            fig1, ax1 = plt.subplots(figsize=(6, 3.5))
            ax1.bar(tab_edad["edad_cat"], tab_edad["Exposición"], color="#4C72B0")
            ax1.set_xlabel("Edad"); ax1.set_ylabel("Años-póliza")
            ax1.spines[["top", "right"]].set_visible(False)
            plt.xticks(rotation=45, ha="right"); fig1.tight_layout()
            st.pyplot(fig1)

    with g2:
        with st.container(border = True):
            st.markdown("**Frecuencia por nivel de un factor**")
            factor = st.selectbox("Factor", ["cobertura", "sexo", "uso", "combustible"], index=0)
            tabf = kpis_por(df, factor)
            fig2, ax2 = plt.subplots(figsize=(6, 3.5))
            ax2.bar(tabf[factor], tabf["Frecuencia"], color="#C44E52")
            ax2.axhline(kpi_frecuencia(df), ls="--", color="gray", label="frecuencia global")
            ax2.set_xlabel(factor); ax2.set_ylabel("Frecuencia")
            ax2.spines[["top", "right"]].set_visible(False)
            ax2.legend(); plt.xticks(rotation=45, ha="right"); fig2.tight_layout()
            st.pyplot(fig2)

    st.divider()

with tab_detalle:
    # --- Tabla de KPIs por segmento -----------------------------------------------
    st.subheader("KPIs por segmento")
    var_tabla = st.selectbox("Segmentar por", ["cobertura", "sexo", "uso", "combustible", "edad_cat"], index=0)
    tabla = kpis_por(df, var_tabla)
    st.dataframe(
        tabla.style.format({
            "Exposición": "{:,.0f}", "Nº sin.": "{:,}",
            "Frecuencia": "{:.4f}", "Severidad": "${:,.0f}", "Prima pura": "${:,.2f}",
        }),
        use_container_width=True, hide_index=True,
    )

    csv = tabla.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Descargar tabla como CSV",
        data=csv,
        file_name=f"tabla_kpis_por_{var_tabla}.csv", mime = "text/csv")   
    
    with st.expander("¿Qué significan estas columnas?"):
        st.write("Frecuencia = Σ siniestros / Σ exposición · Severidad ponderada por "
                 "nº de siniestros · Prima pura = Σ monto / Σ exposición.")

with tab_cotiza:
    st.subheader("Cotizador de prima")
    st.caption("Este cotizador lo contruimos mañana 3 de octubre")
    
    cc1,cc2,cc3 = st.columns(3)
    
    perfil = {
        "cobertura": cc1.selectbox("Cobertura", options=sorted(datos["cobertura"].unique()), index=0),
        "sexo": cc2.selectbox("Sexo", options=sorted(datos["sexo"].unique()), index=0),
        "edad_conductor": cc3.slider("Edad", 18,95,35)
    }
    
    if st.button("Cotizar"):
        st.session_state["perfil"] = perfil
    
    if "perfil" in st.session_state:
        st.info("Perfil guardado: " + str(st.session_state["perfil"]))
        st.metric("Prima pura estimada", "-PENDIENTE.")
    
     