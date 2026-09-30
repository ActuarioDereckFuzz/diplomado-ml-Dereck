import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


st.set_page_config(page_title="Tablero de cartera", page_icon="📊", layout="wide")
BANDAS_EDAD = [17, 30, 35, 45, 50, 55, 60, 95]

def ruta_datos():
    """Devuelve la primera ruta que exista. Evita el clásico FileNotFoundError."""
    for r in ["../../../datos/datos.parquet", 
              "../../datos/datos.parquet",
              "../../Modulo_4/m4t2_sesion1/datos/datos.parquet",
              "../../../Modulo_4/m4t2_sesion1/datos/datos.parquet"]:
        if os.path.exists(r):
            return r
    raise FileNotFoundError("No encuentro datos.parquet. Ponlo en una carpeta 'datos/'.")

@st.cache_data
def cargar_datos():
    df = pd.read_parquet(ruta_datos())
    df["edad_cat"] = pd.cut(df["edad_conductor"], bins=BANDAS_EDAD)
    return df


def kpi_frecuencia(df):
    """Σ siniestros / Σ exposición (ponderada por exposición)."""
    return df['num_siniestros'].sum() / df['exposicion'].sum()


def kpi_severidad_media(df):
    """Ponderada por nº de siniestros, solo sobre pólizas con N>0."""
    con = df[df['num_siniestros'] > 0]
    return np.average(con['severidad'], weights=con['num_siniestros']) if len(con) else np.nan

def kpi_prima_pura(df):
    """Σ monto / Σ exposición  (= frecuencia × severidad)."""
    return df['monto_total'].sum() / df['exposicion'].sum()


def kpis_por(df, var):
    filas = []
    for niv, g in df.groupby(var, observed=True):
        filas.append({var: str(niv), "Exposición": g["exposicion"].sum(),
                      "Frecuencia": kpi_frecuencia(g), "Severidad": kpi_severidad_media(g),
                      "Prima pura": kpi_prima_pura(g)})
    return pd.DataFrame(filas).sort_values("Exposición", ascending=False).reset_index(drop=True)




datos = cargar_datos()

st.title("📊 Tablero de cartera — auto")
st.write("Autor: Eric Daniel Hernandez Jardon")


st.write(f"Cartera cargada: **{len(datos):,}** pólizas · **{datos.shape[1]}** columnas")


# Barra lateral: filtros
st.sidebar.header("Filtros")
sel = {}
for col in ["cobertura", "sexo", "uso"]:
    ops = sorted(datos[col].dropna().unique())
    sel[col] = st.sidebar.multiselect(col.capitalize(), ops, default=ops)

emin, emax = int(datos["edad_conductor"].min()), int(datos["edad_conductor"].max())
edad = st.sidebar.slider("Edad", emin, emax, (emin, emax))
#st.dataframe(datos.head(10))

df = datos.copy()
for col, vals in sel.items():
    df = df[df[col].isin(vals)]
df = df[df["edad_conductor"].between(*edad)]
if len(df) == 0:
    st.warning("Ningún registro con esos filtros."); st.stop()



freq = kpi_frecuencia(df)
sev = kpi_severidad_media(df)
prima_pura = kpi_prima_pura(df)

c1, c2, c3 = st.columns(3)

c1.metric("Frecuencia de la cartera", f"{freq:.4f}")
c2.metric("Severidad media", f"${sev:,.0f}")
c3.metric("Prima pura", f"${prima_pura:,.2f}")


st.divider()
g1,g2 = st.columns(2)
with g1:
    st.markdown("**Exposición por banda de edad**")
    t = kpis_por(df, "edad_cat")
    fig, ax = plt.subplots(figsize=(5,3))
    ax.bar(t["edad_cat"], t["Exposición"],  color="#4C72B0")
    plt.xticks(rotation=45, ha="right");
    st.pyplot(fig, use_container_width=True) 
    
with g2:
    st.markdown("**Frecuencia por factor**")
    factor = st.selectbox("Factor", ["cobertura", "sexo", "uso"])
    t2 = kpis_por(df,factor)
    fig2, ax2 = plt.subplots(figsize=(5,3))
    ax2.bar(t2[factor], t2["Frecuencia"], color="#C44E52")
    ax2.axhline(kpi_frecuencia(df), ls="--", color="gray")
    st.pyplot(fig2, use_container_width=True)
    
    
st.divider()
st.subheader("KPIs por segmento")

segvar = st.selectbox("Segmentar por", ["cobertura", "sexo", "uso", "edad_cat"])
st.dataframe(kpis_por(df, segvar), use_container_width=True, hide_index=True)

st.info("Edita este archivo, guarda, y el navegador te ofrecerá **Rerun**.")