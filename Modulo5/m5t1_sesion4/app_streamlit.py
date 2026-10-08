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
from yaml import warnings

# ─────────────────────────────────────────────────────────────────────────────
# 0 · Configuración de la página
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Tablero de cartera — Diplomado ML Seguros",
                   page_icon="📊", layout="wide")

# Bandas de M4 (mismos cortes que en los GLM del Módulo 4)
BANDAS_EDAD = [17, 30, 35, 45, 50, 55, 60, 95]

# Abiqca tu ruta de trabajo a la carpeta del módulo 5, para que el notebook y la app corran sin errores.
# cd "C:\Users\Eric_Daniel\Documents\Ciencias Cursos\Diplomado\diplomado-ml-seguros\diplomado-ml-seguros\Modulo_5\m5t1_sesion1"
# cd "diplomado-ml-seguros\diplomado-ml-seguros\Modulo_5\m5t1_sesion4"
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
    
    c1,c2,c3, c4 = st.columns(4)
    
    perfil = {
        "edad_conductor": c1.slider("Edad", 18,95,35),
        "cobertura": c1.selectbox("Cobertura", options=sorted(datos["cobertura"].unique()), index=0),
        "sexo": c2.selectbox("Sexo", options=sorted(datos["sexo"].unique()), index=0),
        "uso": c2.selectbox("Uso", options=sorted(datos["uso"].unique()), index=0),
        "combustible": c3.selectbox("Combustible", options=sorted(datos["combustible"].unique()), index=0),
        "antiguedad": c3.slider("Antigüedad del vehículo", 0, 40, 5),
        "potencia": c4.slider("Potencia del vehículo (HP)", 10, 200, 70),
        "bonus": c4.slider("Bonus-malus", 0, 25,5)
    }
    
    st.sidebar.header("Recargos de tarifa")
    gatos = st.sidebar.slider("Gastos", 0.0,0.5,0.25,0.01)
    margen = st.sidebar.slider("Margen", 0.0,0.3,0.05,0.01)
    
    if st.button("Cotizar"):
        st.session_state["perfil"] = perfil
    
    BANDINGS = {"edad_conductor": [17, 30, 35, 45, 50, 55, 60, 95],
               "antiguedad_vehiculo": [-1,1,2, 3,4, 5, 10, 15, 50],
               "potencia": [9,40,50,60,70,250],
               "nivel_bonus": [-1,0,1,2,5,10,25]}
     
    def banda(col, val):
        return str(pd.cut([val], bins = BANDINGS[col])[0])
    
    def cotizar_empirico(d,p):
        seg = d[(d["edad_cat"].astype(str) == banda("edad_conductor", p["edad_conductor"])) &
                (d["cobertura"]==p["cobertura"]) &
                (d["sexo"]==p["sexo"]) &
                (d["uso"]==p["uso"]) &
                (d["combustible"]==p["combustible"]) 
                # &
                # (d["antiguedad_vehiculo"]==p["antiguedad"]) &
                # (d["potencia"]==p["potencia"]) &
                # (d["nivel_bonus"]==p["bonus"])
        ]

        n = len(seg)
        pp = seg["monto_total"].sum() / seg["exposicion"].sum() if seg["exposicion"].sum() > 0 else np.nan
        return pp,n
    

    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    import warnings; warnings.filterwarnings('ignore')
    
    @st.cache_resource
    def ajustar_modelos(d):
        sev = d[d['num_siniestros'] > 0].copy()
        m_freq = smf.glm(
        "num_siniestros ~ C(nivel_bonus_cat)+C(edad_conductor_cat)+C(combustible)+"
        "C(antiguedad_vehiculo_cat)+C(potencia_cat)+C(cobertura)+C(uso)",
        d, family=sm.families.Poisson(), offset=np.log(d["exposicion"])).fit()
        m_sev = smf.glm(
            "severidad ~ C(edad_conductor_cat)+C(antiguedad_vehiculo_cat)+C(potencia_cat)+"
            "C(nivel_bonus_cat)+C(sexo)",
            sev, family=sm.families.Gamma(sm.families.links.Log()),
            freq_weights=sev["num_siniestros"]).fit()
        
        d["pure_premium"] = d["monto_total"] / d["exposicion"]
        m_agg = smf.glm("pure_premium ~ C(cobertura)+C(sexo)+C(combustible)+C(edad_conductor_cat)+"
        "C(potencia_cat)+C(nivel_bonus_cat)",
        d, family=sm.families.Tweedie(var_power=1.3, link=sm.families.links.Log()),
        var_weights=d["exposicion"]).fit()
        return m_freq, m_sev, m_agg
    
    for col, cortes in BANDINGS.items():
        datos[col + "_cat"] = pd.cut(datos[col], bins=cortes).astype(str)
        
    m_freq, m_sev , m_agg = ajustar_modelos(datos)
    
    
    def cotizar_glm(p,m_freq,m_sev,m_agg):
        row = pd.DataFrame([
            {"edad_conductor_cat": banda("edad_conductor",p["edad_conductor"]),
            "antiguedad_vehiculo_cat": banda("antiguedad_vehiculo",p["antiguedad"]),
            "potencia_cat": banda("potencia",p["potencia"]),
            "nivel_bonus_cat": banda("nivel_bonus",p["bonus"]),
            "cobertura": p["cobertura"],
            "sexo": p["sexo"],
            "uso": p["uso"],
            "combustible": p["combustible"],
            "exposicion": 1.0}])
        freq = float(m_freq.predict(row, offset=np.log(row["exposicion"]))[0]) 
        sev = float(m_sev.predict(row)[0])
        pp_tw = float(m_agg.predict(row)[0])
        return freq, sev, freq * sev, pp_tw
    
    
    NOMBRE = {"edad_conductor_cat":"edad", "nivel_bonus_cat":"nivel de bonus", "potencia_cat":"potencia",
          "antiguedad_vehiculo_cat":"antigüedad del vehículo", "cobertura":"cobertura",
          "sexo":"sexo", "combustible":"combustible", "uso":"uso"}

    def relatividades(p, m_freq, m_sev):
        def rel(m, pref, niv):                      # factor del nivel (1.0 si es la referencia)
            return float(np.exp(m.params.get(f"C({pref})[T.{niv}]", 0.0)))
        nivs = {"edad_conductor_cat": banda("edad_conductor", p["edad_conductor"]),
                "nivel_bonus_cat":    banda("nivel_bonus", p["bonus"]),
                "potencia_cat":       banda("potencia", p["potencia"]),
                "antiguedad_vehiculo_cat": banda("antiguedad_vehiculo", p["antiguedad"]),
                "cobertura": p["cobertura"], "sexo": p["sexo"],
                "combustible": p["combustible"], "uso": p["uso"]}
        out = [(NOMBRE[pref], str(niv), rel(m_freq, pref, niv) * rel(m_sev, pref, niv))
               for pref, niv in nivs.items()]
        return sorted(out, key=lambda x: abs(np.log(x[2])), reverse=True)   # mayor impacto primero
        
    
    if "perfil" in st.session_state:
        p = st.session_state["perfil"]
        st.info("Perfil guardado: " + str(p))
        
        # Empiriva
        pp_emp, n = cotizar_empirico(datos, p) 
          
        with st.container(border=True):
            st.markdown("**Cotización empírica**")
            if n < 100:
                st.warning(f"Solo {n} pólizas en este segmento. La prima empírica puede ser poco creible.")
            st.metric("Prima pura empírica", "-" if np.isnan(pp_emp) else f"${pp_emp:,.2f}", delta = f"{n} pólizas en el segmento")

        # GLM

        freq, sev, pp_glm, pp_tw = cotizar_glm(p, m_freq, m_sev, m_agg)
        with st.container(border=True):
            st.markdown("**Cotización por GLM**")
            st.metric("Prima pura (freq x sev)", f"${pp_glm:,.2f}")
            st.caption(f"Frecuencia predicha: {freq:.4f} · Severidad predicha: ${sev:,.0f}")
            st.metric("Prima pura (Tweedie)", f"${pp_tw:,.2f}")
            
        tarifa = pp_glm / (1 - gatos - margen)
        st.metric("Prima de TARIFA", f"${tarifa:,.2f}")

        st.subheader("¿Por qué esta prima?")
        rels = relatividades(p, m_freq, m_sev)
        pp_ref = pp_glm / np.prod([r for _, _, r in rels])      # prima del perfil de referencia
        st.caption(f"Prima base: ${pp_ref:,.2f}. Tu perfil la ajusta así:")
        for nombre, niv, r in rels:
            pct = (r - 1) * 100
            if abs(pct) < 3:
                st.write(f"• **{nombre}** ({niv}): sin efecto")
            elif pct > 0:
                st.write(f"• ⬆️ **{nombre}** ({niv}): pagas **+{pct:.0f}%**")
            else:
                st.write(f"• ⬇️ **{nombre}** ({niv}): pagas **{pct:.0f}%**")