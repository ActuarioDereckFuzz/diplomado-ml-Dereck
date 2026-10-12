"""
app_streamlit.py — Tablero de cartera · Módulo 5 · Tema 1  ·  VERSIÓN PARA DEPLOY
================================================================================
Diplomado Machine Learning en Seguros · FC UNAM
Se corre local:   streamlit run app_streamlit.py
Deploy:           Streamlit Community Cloud (ver guía de deploy)

Es la MISMA app que construimos en clase. Solo cambia lo que el deploy exige:
  1) Las rutas se anclan a la carpeta del archivo (AQUI) — en el servidor el
     directorio de trabajo es la RAÍZ del repo, no la carpeta de la app, así que
     "cartera.db" y "datos/datos.parquet" se romperían sin esto.
  2) cargar_datos() lee de SQLite y, si la BD no existe (en el servidor no la
     subimos), la RECONSTRUYE desde el parquet. Por eso el parquet SÍ va al repo
     y cartera.db va en .gitignore.
  3) El cotizador muestra los KPIs en orden (1 empírico → 2 GLM → 3 precio).
"""

import os
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings; warnings.filterwarnings("ignore")

st.set_page_config(page_title="Tablero de cartera — Diplomado ML Seguros",
                   page_icon="📊", layout="wide")

# --- rutas ancladas a ESTA carpeta (clave para el deploy) ---------------------
AQUI    = os.path.dirname(os.path.abspath(__file__))
RUTA_DB = os.path.join(AQUI, "cartera.db")

def ruta_parquet():
    """Busca el parquet SIEMPRE relativo a la carpeta de la app."""
    for r in [os.path.join(AQUI, "datos", "datos.parquet"),
              os.path.join(AQUI, "..", "..", "Modulo4", "m4t2_sesion1", "datos", "datos.parquet")]:
        if os.path.exists(r):
            return r
    raise FileNotFoundError("No encuentro datos.parquet (ponlo en datos/datos.parquet).")

BANDAS_EDAD = [17, 30, 35, 45, 50, 55, 60, 95]
BANDINGS = {"edad_conductor":      [17, 30, 35, 45, 50, 55, 60, 95],
            "antiguedad_vehiculo": [-1, 1, 2, 3, 4, 5, 10, 15, 50],
            "potencia":            [9, 40, 50, 60, 70, 250],
            "nivel_bonus":         [-1, 0, 1, 2, 5, 10, 25]}


# ─────────────────────────────────────────────────────────────────────────────
# 1 · Carga de datos — desde SQLite, con red de seguridad para el deploy
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def cargar_datos():
    # si no existe la BD o la tabla, se construye desde el parquet
    crear_tabla = False
    if not os.path.exists(RUTA_DB):
        crear_tabla = True
    else:
        con0 = sqlite3.connect(RUTA_DB)
        tablas = con0.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        if ("cartera",) not in tablas:
            crear_tabla = True
        con0.close()

    if crear_tabla:
        con0 = sqlite3.connect(RUTA_DB)
        pd.read_parquet(ruta_parquet()).to_sql("cartera", con0, if_exists="replace", index=False)
        con0.close()

    con = sqlite3.connect(RUTA_DB)
    df = pd.read_sql("SELECT * FROM cartera", con)
    con.close()

    # agrega la columna categórica de edad
    if "edad_conductor" in df.columns:
        df["edad_cat"] = pd.cut(df["edad_conductor"], bins=BANDAS_EDAD)
    return df



# ─────────────────────────────────────────────────────────────────────────────
# 2 · KPIs
# ─────────────────────────────────────────────────────────────────────────────
def kpi_exposicion(df):      return df["exposicion"].sum()
def kpi_num_siniestros(df):  return int(df["num_siniestros"].sum())
def kpi_frecuencia(df):      return df["num_siniestros"].sum() / df["exposicion"].sum()
def kpi_severidad_media(df):
    con = df[df["num_siniestros"] > 0]
    if con["num_siniestros"].sum() == 0: return np.nan
    return np.average(con["severidad"], weights=con["num_siniestros"])
def kpi_prima_pura(df):      return df["monto_total"].sum() / df["exposicion"].sum()

def kpis_por(df, variable):
    filas = []
    for nivel, g in df.groupby(variable, observed=True):
        filas.append({variable: str(nivel), "Exposición": kpi_exposicion(g),
                      "Nº sin.": kpi_num_siniestros(g), "Frecuencia": kpi_frecuencia(g),
                      "Severidad": kpi_severidad_media(g), "Prima pura": kpi_prima_pura(g)})
    return pd.DataFrame(filas).sort_values("Exposición", ascending=False).reset_index(drop=True)

def ratios(sub, gastos, margen):
    """Loss ratio y combined ratio del segmento (usa la prima de tarifa del GLM)."""
    primas = (sub["prima_pura_glm"] / (1 - gastos - margen)).sum()
    if primas == 0: return np.nan, np.nan
    loss = sub["monto_total"].sum() / primas
    return loss, loss + gastos


# ─────────────────────────────────────────────────────────────────────────────
# 3 · Modelos GLM (setup compartido por Cartera y Cotizador)
# ─────────────────────────────────────────────────────────────────────────────
def banda(col, val):
    return str(pd.cut([val], bins=BANDINGS[col])[0])

@st.cache_resource
def ajustar_modelos(d):
    sev = d[d["num_siniestros"] > 0].copy()
    m_freq = smf.glm("num_siniestros ~ C(nivel_bonus_cat)+C(edad_conductor_cat)+C(combustible)+"
                     "C(antiguedad_vehiculo_cat)+C(potencia_cat)+C(cobertura)+C(uso)",
                     d, family=sm.families.Poisson(), offset=np.log(d["exposicion"])).fit()
    m_sev = smf.glm("severidad ~ C(edad_conductor_cat)+C(antiguedad_vehiculo_cat)+C(potencia_cat)+"
                    "C(nivel_bonus_cat)+C(sexo)",
                    sev, family=sm.families.Gamma(sm.families.links.Log()),
                    freq_weights=sev["num_siniestros"]).fit()
    d["pure_premium"] = d["monto_total"] / d["exposicion"]
    m_agg = smf.glm("pure_premium ~ C(cobertura)+C(sexo)+C(combustible)+C(edad_conductor_cat)+"
                    "C(potencia_cat)+C(nivel_bonus_cat)",
                    d, family=sm.families.Tweedie(var_power=1.3, link=sm.families.links.Log()),
                    var_weights=d["exposicion"]).fit()
    return m_freq, m_sev, m_agg

@st.cache_data
def agregar_prima_glm(datos, _m_freq, _m_sev):
    """Prima pura del GLM por póliza (pérdida esperada) — para los ratios de rentabilidad."""
    out = datos.copy()
    out["prima_pura_glm"] = (_m_freq.predict(out, offset=np.log(out["exposicion"]))
                             * _m_sev.predict(out))
    return out

NOMBRE = {"edad_conductor_cat":"edad", "nivel_bonus_cat":"nivel de bonus", "potencia_cat":"potencia",
          "antiguedad_vehiculo_cat":"antigüedad", "cobertura":"cobertura", "sexo":"sexo",
          "combustible":"combustible", "uso":"uso"}

def cotizar_empirico(d, p):
    # el .astype(str) es la corrección: edad_cat es Interval y hay que compararlo como texto
    seg = d[(d["edad_cat"].astype(str) == banda("edad_conductor", p["edad_conductor"])) &
            (d["cobertura"] == p["cobertura"]) & (d["sexo"] == p["sexo"]) &
            (d["uso"] == p["uso"]) & (d["combustible"] == p["combustible"])
            # TAREA alumnos: agregar aquí antiguedad, potencia y bonus (banda al vuelo)
            ]
    n = len(seg)
    pp = seg["monto_total"].sum() / seg["exposicion"].sum() if seg["exposicion"].sum() > 0 else np.nan
    return pp, n

def cotizar_glm(p, m_freq, m_sev, m_agg):
    row = pd.DataFrame([{"edad_conductor_cat": banda("edad_conductor", p["edad_conductor"]),
                         "antiguedad_vehiculo_cat": banda("antiguedad_vehiculo", p["antiguedad"]),
                         "potencia_cat": banda("potencia", p["potencia"]),
                         "nivel_bonus_cat": banda("nivel_bonus", p["bonus"]),
                         "cobertura": p["cobertura"], "sexo": p["sexo"], "uso": p["uso"],
                         "combustible": p["combustible"], "exposicion": 1.0}])
    freq = float(m_freq.predict(row, offset=np.log(row["exposicion"]))[0])
    sev  = float(m_sev.predict(row)[0])
    pp_tw = float(m_agg.predict(row)[0])
    return freq, sev, freq * sev, pp_tw

def relatividades(p, m_freq, m_sev):
    def rel(m, pref, niv): return float(np.exp(m.params.get(f"C({pref})[T.{niv}]", 0.0)))
    nivs = {"edad_conductor_cat": banda("edad_conductor", p["edad_conductor"]),
            "nivel_bonus_cat": banda("nivel_bonus", p["bonus"]),
            "potencia_cat": banda("potencia", p["potencia"]),
            "antiguedad_vehiculo_cat": banda("antiguedad_vehiculo", p["antiguedad"]),
            "cobertura": p["cobertura"], "sexo": p["sexo"],
            "combustible": p["combustible"], "uso": p["uso"]}
    out = [(NOMBRE[pref], str(niv), rel(m_freq, pref, niv) * rel(m_sev, pref, niv))
           for pref, niv in nivs.items()]
    return sorted(out, key=lambda x: abs(np.log(x[2])), reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4 · Preparar datos + modelos (una sola vez, ANTES de los tabs)
# ─────────────────────────────────────────────────────────────────────────────
datos = cargar_datos()
for col, cortes in BANDINGS.items():
    datos[col + "_cat"] = pd.cut(datos[col], bins=cortes).astype(str)
m_freq, m_sev, m_agg = ajustar_modelos(datos)
datos = agregar_prima_glm(datos, m_freq, m_sev)

st.title("📊 Tablero de cartera — auto")

# --- Barra lateral: filtros ---
st.sidebar.header("Filtros")
with st.sidebar.form("filtros"):
    cob  = st.multiselect("Cobertura", sorted(datos["cobertura"].unique()), sorted(datos["cobertura"].unique()))
    sexo = st.multiselect("Sexo", sorted(datos["sexo"].unique()), sorted(datos["sexo"].unique()))
    uso  = st.multiselect("Uso", sorted(datos["uso"].unique()), sorted(datos["uso"].unique()))
    comb = st.multiselect("Combustible", sorted(datos["combustible"].unique()), sorted(datos["combustible"].unique()))
    emin, emax = int(datos["edad_conductor"].min()), int(datos["edad_conductor"].max())
    edad = st.slider("Edad", emin, emax, (emin, emax))
    st.form_submit_button("Aplicar filtros")

# --- Recargos de tarifa (compartidos por Cartera y Cotizador) ---
st.sidebar.header("Recargos de tarifa")
gastos = st.sidebar.slider("Gastos", 0.0, 0.5, 0.25, 0.01)
margen = st.sidebar.slider("Margen", 0.0, 0.3, 0.05, 0.01)

df = datos[datos["cobertura"].isin(cob) & datos["sexo"].isin(sexo) &
           datos["uso"].isin(uso) & datos["combustible"].isin(comb)]
df = df[df["edad_conductor"].between(*edad)]
if len(df) == 0:
    st.warning("Ningún registro con esos filtros."); st.stop()
st.sidebar.metric("Pólizas seleccionadas", f"{len(df):,}")

tab_cartera, tab_detalle, tab_cotiza = st.tabs(["Cartera", "Detalle", "Cotizador"])

# ============================ CARTERA ============================
with tab_cartera:
    st.subheader("Indicadores de la selección")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Exposición", f"{kpi_exposicion(df):,.0f}")
    c2.metric("Nº siniestros", f"{kpi_num_siniestros(df):,}")
    c3.metric("Frecuencia", f"{kpi_frecuencia(df):.4f}",
              delta=f"{kpi_frecuencia(df)-kpi_frecuencia(datos):+.4f} vs cartera")
    c4.metric("Severidad media", f"${kpi_severidad_media(df):,.0f}",
              delta=f"${kpi_severidad_media(df)-kpi_severidad_media(datos):+,.0f}")
    c5.metric("Prima pura", f"${kpi_prima_pura(df):,.2f}")

    # --- Rentabilidad (ratios) ---
    st.subheader("Rentabilidad")
    loss, comb = ratios(df, gastos, margen)
    r1, r2 = st.columns(2)
    r1.metric("Loss ratio", f"{loss:.1%}")
    r2.metric("Combined ratio", f"{comb:.1%}",
              delta=("rentable" if comb < 1 else "pierde"),
              delta_color=("normal" if comb < 1 else "inverse"))

    st.divider()
    g1, g2 = st.columns(2)
    with g1:
        with st.container(border=True):
            st.markdown("**Frecuencia por factor**")
            factor = st.selectbox("Factor", ["cobertura", "sexo", "uso", "combustible"], index=0)
            tabf = kpis_por(df, factor)
            fig, ax = plt.subplots(figsize=(6, 3.2))
            ax.bar(tabf[factor], tabf["Frecuencia"], color="#C44E52")
            ax.axhline(kpi_frecuencia(df), ls="--", color="gray")
            ax.spines[["top", "right"]].set_visible(False)
            plt.xticks(rotation=45, ha="right"); fig.tight_layout(); st.pyplot(fig)
    with g2:
        with st.container(border=True):
            st.markdown(f"**Combined ratio por {factor}** (línea roja = 100%)")
            rows = [(str(niv), ratios(g, gastos, margen)[1]) for niv, g in df.groupby(factor, observed=True)]
            fig2, ax2 = plt.subplots(figsize=(6, 3.2))
            cols = ["#C44E52" if c >= 1 else "#1C7293" for _, c in rows]
            ax2.bar([r[0] for r in rows], [r[1] for r in rows], color=cols)
            ax2.axhline(1.0, ls="--", color="red")
            ax2.spines[["top", "right"]].set_visible(False)
            plt.xticks(rotation=45, ha="right"); fig2.tight_layout(); st.pyplot(fig2)

# ============================ DETALLE ============================
with tab_detalle:
    st.subheader("KPIs por segmento")
    var_tabla = st.selectbox("Segmentar por", ["cobertura", "sexo", "uso", "combustible", "edad_cat"], index=0)
    tabla = kpis_por(df, var_tabla)
    st.dataframe(tabla.style.format({"Exposición": "{:,.0f}", "Nº sin.": "{:,}",
                 "Frecuencia": "{:.4f}", "Severidad": "${:,.0f}", "Prima pura": "${:,.2f}"}),
                 use_container_width=True, hide_index=True)
    st.download_button("Descargar tabla como CSV", tabla.to_csv(index=False).encode("utf-8"),
                       f"tabla_kpis_por_{var_tabla}.csv", "text/csv")
    with st.expander("¿Qué significan estas columnas?"):
        st.write("Frecuencia = Σ siniestros / Σ exposición · Severidad ponderada por nº de "
                 "siniestros · Prima pura = Σ monto / Σ exposición.")

# ============================ COTIZADOR ============================
with tab_cotiza:
    st.subheader("Cotizador de prima")
    c1, c2, c3, c4 = st.columns(4)
    perfil = {"edad_conductor": c1.slider("Edad", 18, 95, 35),
              "cobertura": c1.selectbox("Cobertura", sorted(datos["cobertura"].unique())),
              "sexo": c2.selectbox("Sexo", sorted(datos["sexo"].unique())),
              "uso": c2.selectbox("Uso", sorted(datos["uso"].unique())),
              "combustible": c3.selectbox("Combustible", sorted(datos["combustible"].unique())),
              "antiguedad": c3.slider("Antigüedad del vehículo", 0, 40, 5),
              "potencia": c4.slider("Potencia (HP)", 10, 200, 70),
              "bonus": c4.slider("Bonus-malus", 0, 25, 5)}
    if st.button("Cotizar", type="primary"):
        st.session_state["perfil"] = perfil

    if "perfil" in st.session_state:
        p = st.session_state["perfil"]
        pp_emp, n = cotizar_empirico(datos, p)
        freq, sev, pp_glm, pp_tw = cotizar_glm(p, m_freq, m_sev, m_agg)
        tarifa = pp_glm / (1 - gastos - margen)

        # KPIs del cotizador EN ORDEN: 1 lo que dice la data → 2 el modelo → 3 el precio
        k1, k2, k3 = st.columns(3)
        with k1:
            with st.container(border=True):
                st.caption("1 · Empírico (segmento)")
                st.metric("Prima pura", "—" if np.isnan(pp_emp) else f"${pp_emp:,.2f}")
                st.caption(f"{n:,} pólizas" + ("  ⚠️ poco creíble" if n < 100 else ""))
        with k2:
            with st.container(border=True):
                st.caption("2 · GLM (modelo)")
                st.metric("Prima pura", f"${pp_glm:,.2f}")
                st.caption(f"freq {freq:.4f} · sev ${sev:,.0f} · Tweedie ${pp_tw:,.2f}")
        with k3:
            with st.container(border=True):
                st.caption("3 · Precio al cliente")
                st.metric("Prima de tarifa", f"${tarifa:,.2f}")
                st.caption(f"recargo ×{1/(1-gastos-margen):.3f}")

        # ---- ¿Por qué esta prima? (rating factors, versión gráfica) ----
        st.markdown("#### ¿Por qué esta prima?")
        rels = relatividades(p, m_freq, m_sev)
        pp_ref = pp_glm / np.prod([r for _, _, r in rels])
        etiquetas = [f"{nombre}: {niv}" for nombre, niv, r in rels]
        pct = [(r - 1) * 100 for _, _, r in rels]
        colores = ["#C44E52" if v > 0 else "#2A9D8F" for v in pct]
        fig3, ax3 = plt.subplots(figsize=(8, 0.5 * len(rels) + 0.6))
        ax3.barh(etiquetas, pct, color=colores)
        ax3.axvline(0, color="#555", lw=1)
        ax3.set_xlabel("Ajuste sobre la prima base (%)")
        ax3.spines[["top", "right"]].set_visible(False)
        for y, v in enumerate(pct):
            ax3.text(v + (1 if v >= 0 else -1), y, f"{v:+.0f}%", va="center",
                     ha="left" if v >= 0 else "right", fontsize=9)
        ax3.invert_yaxis()           # mayor impacto arriba
        ax3.margins(x=0.18)          # aire para que las etiquetas no se encimen
        fig3.tight_layout()
        st.pyplot(fig3)
        st.caption(f"Prima base (perfil de referencia): ${pp_ref:,.2f} → tu prima pura: ${pp_glm:,.2f}. "
                   "Rojo = pagas más · verde = pagas menos.")
