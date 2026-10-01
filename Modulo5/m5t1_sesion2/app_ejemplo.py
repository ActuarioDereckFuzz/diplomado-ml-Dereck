from shiny import ui, App, render , reactive 
import pandas as pd
import os
import numpy as np


BANDAS_EDAD = [17, 30, 35, 45, 50, 55, 60, 95]
# cd "diplomado-ml-seguros\diplomado-ml-seguros\Modulo_5\m5t1_sesion2"
# shiny run --reload app_ejemplo.py


## Funciones 

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



# 0 carga de datos

# ─────────────────────────────────────────────────────────────────────────────
# 0 · Carga de datos — en Shiny se hace UNA vez, al arrancar el módulo.
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

# 1) app_ui = LO QUE SE VE (se declara, no se ejecuta de arriba a abajo)

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_select("cobertura", "Cobertura", choices=sorted(datos["cobertura"].unique())),
        ui.input_select("sexo", "Sexo", choices=sorted(datos["sexo"].unique())),
        ui.input_slider("edad", "Edad", min=int(datos["edad_conductor"].min()), max=int(datos["edad_conductor"].max()), value=[int(datos["edad_conductor"].min()), int(datos["edad_conductor"].max())]),
        title="Filtros"),
    ui.h3("Tablero cartera autos"),
    ui.layout_columns(
        ui.value_box("Frecuencia", ui.output_text("vb_freq")),
        ui.value_box("Severidad media", ui.output_text("vb_sev")),
        ui.value_box("Prima pura", ui.output_text("vb_pp")),
    ),
    ui.card(
        ui.card_header("Exposición por banda de edad"),
        ui.output_plot("plot_edad"),
    ),
    ui.card(
        ui.card_header("Frecuencia por factor"),
        ui.input_select("factor", None, choices=["cobertura", "sexo", "uso"]),
        ui.output_plot("plot_factor")),
    ui.card(
        ui.card_header("Tabla de KPIs por factor"),
        ui.output_table("tabla_kpis"),
    ),
    title = "Tablero Shiny"
)
# 2) server = LA LÓGICA (aquí viven inputs, cálculos y outputs)
def server(input, output, session):
    @reactive.Calc
    def df_filtrado():
        d = datos.copy()
        d = d[d["cobertura"] == input.cobertura()]
        d = d[d["sexo"] == input.sexo()]
        emin, emax = input.edad()
        d = d[(d["edad_conductor"] >= emin) & (d["edad_conductor"] <= emax)]
        return d

    
    @render.text
    def vb_freq():
        d = df_filtrado()
        freq = kpi_frecuencia(d)
        return f"{freq:.4f}"
    
    @render.text
    def vb_sev():
        d = df_filtrado()
        sev = kpi_severidad_media(d)
        return f"{sev:,.0f}" if not np.isnan(sev) else "N/A"

    @render.text
    def vb_pp():
        d = df_filtrado()
        pp = kpi_prima_pura(d)
        return f"{pp:,.2f}" if not np.isnan(pp) else "N/A"

    @render.plot
    def plot_edad():
        d = df_filtrado()
        kpis = kpis_por(d, "edad_cat")
        ax = kpis.plot.bar(x="edad_cat", y="Exposición", legend=False, rot=0)
        ax.set_ylabel("Exposición")
        return ax.figure

    @render.plot
    def plot_factor():
        d = df_filtrado()
        factor = input.factor()
        kpis = kpis_por(d, factor)
        ax = kpis.plot.bar(x=factor, y="Frecuencia", legend=False, rot=0)
        ax.set_ylabel("Frecuencia")
        return ax.figure

    @render.data_frame
    def tabla_kpis():
        return render.DataGrid(kpis_por(df_filtrado(), "cobertura"), width="100%", height="300px")

# 3) App = une las dos piezas
app = App(app_ui, server)
