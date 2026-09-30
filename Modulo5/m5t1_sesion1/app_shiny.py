"""
app_shiny.py — El MISMO tablero, ahora en Shiny for Python
==========================================================
Diplomado Machine Learning en Seguros · FC UNAM · Módulo 5 · Tema 1 · Sesión 1

Este archivo es el "gemelo" de app_streamlit.py. Mismos datos, mismos KPIs, mismos
gráficos — pero escrito en Shiny for Python. Sirve para mostrar EN CLASE en qué se
parecen y en qué se diferencian los dos frameworks.

──────────────────────────────────────────────────────────────────────────────
QUÉ SE MUEVE respecto a Streamlit (esto es lo que hay que señalar en la sesión):

  Streamlit                              Shiny for Python
  ─────────────────────────────────     ─────────────────────────────────────────
  El script se RE-EJECUTA completo       Declaras UI (app_ui) y lógica (server) por
  en cada interacción.                   separado; NADA se re-ejecuta completo.

  Lees el valor de un widget con         Lees un input con input.<id>()  → es una
  la variable que devuelve el widget.    función que "avisa" cuando cambia.

  Filtras el df en el cuerpo del         Filtras en un @reactive.calc: se recalcula
  script; todo lo de abajo se recalcula. SOLO cuando cambia un input del que depende.

  st.metric / st.pyplot / st.dataframe   ui.value_box + @render.text / @render.plot /
  se llaman en orden, de arriba a abajo. @render.data_frame, enlazados por id.

  Idea: "re-ejecuto todo".               Idea: "declaro qué depende de qué" (reactivo).
──────────────────────────────────────────────────────────────────────────────

Cómo se corre (NO con python, igual que Streamlit necesita su runtime):
    conda activate diplomado
    pip install shiny            # una sola vez, si no lo tienes
    cd "diplomado-ml-seguros\diplomado-ml-seguros\Modulo_5\m5t1_sesion1" #ubiuca la ruta de trabajo
    shiny run --reload app_shiny.py
    # o, sin activar el entorno, desde VS Code:
    #   & "C:\\Users\\Eric_Daniel\\anaconda3\\envs\\diplomado\\python.exe" -m shiny run --reload app_shiny.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shiny import App, reactive, render, ui

BANDAS_EDAD = [17, 30, 35, 45, 50, 55, 60, 95]


# ─────────────────────────────────────────────────────────────────────────────
# 1 · Carga de datos — en Shiny se hace UNA vez, al arrancar el módulo.
#     (No hay rerun del script como en Streamlit, así que no necesitamos cache.)
# ─────────────────────────────────────────────────────────────────────────────
def cargar_datos():
    candidatos = [
        "datos/datos.parquet", "datos/datos.pkl",
        "../../Modulo4/m4t2_sesion1/datos/datos.parquet",
        "../../Modulo4/m4t2_sesion1/datos/datos.pkl",
    ]
    ruta = next((r for r in candidatos if os.path.exists(r)), None)
    if ruta is None:
        raise FileNotFoundError("No encuentro datos.parquet ni datos.pkl (cartera de M4).")
    df = pd.read_parquet(ruta) if ruta.endswith(".parquet") else pd.read_pickle(ruta)
    df["edad_cat"] = pd.cut(df["edad_conductor"], bins=BANDAS_EDAD)
    return df


datos = cargar_datos()


# ─────────────────────────────────────────────────────────────────────────────
# 2 · KPIs — LAS MISMAS funciones del notebook y del app de Streamlit
# ─────────────────────────────────────────────────────────────────────────────
def kpi_exposicion(df):      return df["exposicion"].sum()
def kpi_num_siniestros(df):  return int(df["num_siniestros"].sum())
def kpi_frecuencia(df):      return df["num_siniestros"].sum() / df["exposicion"].sum()

def kpi_severidad_media(df):
    con = df[df["num_siniestros"] > 0]
    if con["num_siniestros"].sum() == 0:
        return np.nan
    return np.average(con["severidad"], weights=con["num_siniestros"])

def kpi_prima_pura(df):      return df["monto_total"].sum() / df["exposicion"].sum()


def kpis_por(df, variable):
    filas = []
    for nivel, g in df.groupby(variable, observed=True):
        filas.append({
            variable: str(nivel),
            "Exposición": round(kpi_exposicion(g)),
            "Nº sin.": kpi_num_siniestros(g),
            "Frecuencia": round(kpi_frecuencia(g), 4),
            "Severidad": round(kpi_severidad_media(g)),
            "Prima pura": round(kpi_prima_pura(g), 2),
        })
    return pd.DataFrame(filas).sort_values("Exposición", ascending=False).reset_index(drop=True)


# opciones de filtros (a partir de los valores reales)
def _ops(col): return sorted(datos[col].dropna().unique().tolist())
EMIN, EMAX = int(datos["edad_conductor"].min()), int(datos["edad_conductor"].max())


# ─────────────────────────────────────────────────────────────────────────────
# 3 · UI — se DECLARA (no se ejecuta de arriba a abajo como en Streamlit)
# ─────────────────────────────────────────────────────────────────────────────
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_selectize("cobertura", "Cobertura", {c: c for c in _ops("cobertura")},
                           multiple=True, selected=_ops("cobertura")),
        ui.input_selectize("sexo", "Sexo", {c: c for c in _ops("sexo")},
                           multiple=True, selected=_ops("sexo")),
        ui.input_selectize("uso", "Uso", {c: c for c in _ops("uso")},
                           multiple=True, selected=_ops("uso")),
        ui.input_selectize("combustible", "Combustible", {c: c for c in _ops("combustible")},
                           multiple=True, selected=_ops("combustible")),
        ui.input_slider("edad", "Edad del conductor", min=EMIN, max=EMAX, value=(EMIN, EMAX)),
        title="Filtros",
    ),
    ui.h4("Indicadores de la selección"),
    ui.layout_columns(
        ui.value_box("Exposición", ui.output_text("vb_expo")),
        ui.value_box("Nº siniestros", ui.output_text("vb_nsin")),
        ui.value_box("Frecuencia", ui.output_text("vb_freq")),
        ui.value_box("Severidad media", ui.output_text("vb_sev")),
        ui.value_box("Prima pura", ui.output_text("vb_pp")),
        col_widths=[2, 2, 2, 3, 3],
    ),
    ui.layout_columns(
        ui.card(ui.card_header("Exposición por banda de edad"), ui.output_plot("plot_edad")),
        ui.card(
            ui.card_header("Frecuencia por factor"),
            ui.input_select("factor", None, ["cobertura", "sexo", "uso", "combustible"]),
            ui.output_plot("plot_factor"),
        ),
    ),
    ui.card(
        ui.card_header("KPIs por segmento"),
        ui.input_select("segvar", "Segmentar por",
                        ["cobertura", "sexo", "uso", "combustible", "edad_cat"]),
        ui.output_data_frame("tabla"),
    ),
    title="Tablero de cartera — auto (Shiny for Python)",
)


# ─────────────────────────────────────────────────────────────────────────────
# 4 · SERVER — la lógica reactiva
# ─────────────────────────────────────────────────────────────────────────────
def server(input, output, session):

    @reactive.calc
    def df_filtrado():
        """Se recalcula SOLO cuando cambia un input del que depende (no en cada 'rerun')."""
        d = datos
        for col in ["cobertura", "sexo", "uso", "combustible"]:
            sel = input[col]()
            if sel:  # si no hay selección, se interpreta como 'todos'
                d = d[d[col].isin(sel)]
        lo, hi = input.edad()
        return d[d["edad_conductor"].between(lo, hi)]

    # --- Tarjetas (value boxes) ---
    @render.text
    def vb_expo(): return f"{kpi_exposicion(df_filtrado()):,.0f}"
    @render.text
    def vb_nsin(): return f"{kpi_num_siniestros(df_filtrado()):,}"
    @render.text
    def vb_freq(): return f"{kpi_frecuencia(df_filtrado()):.4f}"
    @render.text
    def vb_sev():
        s = kpi_severidad_media(df_filtrado())
        return "—" if np.isnan(s) else f"${s:,.0f}"
    @render.text
    def vb_pp(): return f"${kpi_prima_pura(df_filtrado()):,.2f}"

    # --- Gráficos ---
    @render.plot
    def plot_edad():
        tab = kpis_por(df_filtrado(), "edad_cat")
        fig, ax = plt.subplots()
        ax.bar(tab["edad_cat"], tab["Exposición"], color="#4C72B0")
        ax.set_xlabel("Edad"); ax.set_ylabel("Años-póliza")
        ax.spines[["top", "right"]].set_visible(False)
        for t in ax.get_xticklabels():
            t.set_rotation(45); t.set_ha("right")
        return fig

    @render.plot
    def plot_factor():
        var = input.factor()
        tab = kpis_por(df_filtrado(), var)
        fig, ax = plt.subplots()
        ax.bar(tab[var], tab["Frecuencia"], color="#C44E52")
        ax.axhline(kpi_frecuencia(df_filtrado()), ls="--", color="gray", label="frecuencia global")
        ax.set_xlabel(var); ax.set_ylabel("Frecuencia")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend()
        for t in ax.get_xticklabels():
            t.set_rotation(45); t.set_ha("right")
        return fig

    # --- Tabla ---
    @render.data_frame
    def tabla():
        return render.DataGrid(kpis_por(df_filtrado(), input.segvar()), width="100%")


app = App(app_ui, server)
