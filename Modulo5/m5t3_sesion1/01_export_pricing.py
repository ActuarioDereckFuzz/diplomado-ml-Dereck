"""
01_export_pricing.py  —  De los coeficientes del GLM a la TARIFA en Excel
=========================================================================
Módulo 5 · Tema 2 · Parte 3 · MIÉRCOLES (pricing)

Idea: el modelo lo construimos en Python; el entregable de tarifa va en Excel.
Tomamos el GLM agregado (Tweedie, el mismo de Módulo 4) y escribimos:
  - una PORTADA/resumen,
  - el MANUAL DE TARIFA (prima base + factores por variable),
  - un COTIZADOR VIVO en Excel (dropdowns + fórmulas; funciona SIN Python),
  - una hoja de CONTROL que reconcilia Python vs Excel (deben coincidir),
  - un REPRECIO de cartera con su impacto.

Librería clave: xlsxwriter (crear desde cero, con formato, fórmulas y validación).
Se corre:  python 01_export_pricing.py   ->   genera  tarifa_cotizador.xlsx
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings; warnings.filterwarnings("ignore")
#pip install XlsxWriter
from xlsxwriter.utility import xl_rowcol_to_cell as cell, xl_range_abs as rng_abs

#cd "diplomado-ml-seguros/diplomado-ml-seguros/Modulo_5/m5t2.3_sesion1/"   
# ──────────────────────────────────────────────────────────────────────────
# 0 · Datos + modelo  (esto YA lo tienen de Módulo 4; aquí solo lo recuperamos)
# ──────────────────────────────────────────────────────────────────────────
BINS = {"edad_conductor": [17, 30, 35, 45, 50, 55, 60, 95],
        "potencia":       [9, 40, 50, 60, 70, 250],
        "nivel_bonus":    [-1, 0, 1, 2, 5, 10, 25]}

datos = pd.read_parquet("datos/datos.parquet")
for col, cortes in BINS.items():
    datos[col + "_cat"] = pd.cut(datos[col], bins=cortes).astype(str)
datos["pure_premium"] = datos["monto_total"] / datos["exposicion"]

# GLM agregado (Tweedie, log): prima pura multiplicativa = base × factores
m_agg = smf.glm("pure_premium ~ C(cobertura)+C(sexo)+C(combustible)+"
                "C(edad_conductor_cat)+C(potencia_cat)+C(nivel_bonus_cat)",
                datos, family=sm.families.Tweedie(var_power=1.3, link=sm.families.links.Log()),
                var_weights=datos["exposicion"]).fit()

# ──────────────────────────────────────────────────────────────────────────
# 1 · De coeficientes a TARIFA: base = exp(intercepto), factor = exp(coef)
#     (así la prima pura = base × producto de factores: una tarifa multiplicativa)
# ──────────────────────────────────────────────────────────────────────────
PRIMA_BASE = float(np.exp(m_agg.params["Intercept"]))

VARIABLES = ["cobertura", "sexo", "combustible",
             "edad_conductor_cat", "potencia_cat", "nivel_bonus_cat"]
NOMBRE = {"cobertura":"Cobertura", "sexo":"Sexo", "combustible":"Combustible",
          "edad_conductor_cat":"Edad (banda)", "potencia_cat":"Potencia (banda)",
          "nivel_bonus_cat":"Bonus (banda)"}

def tabla_factores(var):
    """(nivel, factor) de una variable. El nivel de referencia tiene factor = 1."""
    niveles = sorted(datos[var].dropna().unique().tolist())
    filas = [(str(niv), float(np.exp(m_agg.params.get(f"C({var})[T.{niv}]", 0.0))))
             for niv in niveles]
    return filas

FACTORES = {v: tabla_factores(v) for v in VARIABLES}

def prima_python(perfil):
    """Prima pura que da el modelo = base × Π factores (para reconciliar con Excel)."""
    f = PRIMA_BASE
    for v in VARIABLES:
        f *= dict(FACTORES[v])[str(perfil[v])]
    return f

# ──────────────────────────────────────────────────────────────────────────
# 2 · Escribir el Excel con xlsxwriter
# ──────────────────────────────────────────────────────────────────────────
wb = pd.ExcelWriter("tarifa_cotizador.xlsx", engine="xlsxwriter")
book = wb.book

# --- formatos reutilizables ---
F = dict(
    titulo = book.add_format({"bold": True, "font_size": 16, "font_color": "#0F2A43"}),
    sub    = book.add_format({"italic": True, "font_color": "#6B7C8A"}),
    head   = book.add_format({"bold": True, "font_color": "white", "bg_color": "#1C7293",
                              "border": 1, "align": "center", "valign": "vcenter"}),
    txt    = book.add_format({"border": 1}),
    money  = book.add_format({"num_format": "$#,##0.00", "border": 1}),
    money_b= book.add_format({"num_format": "$#,##0.00", "border": 1, "bold": True,
                              "bg_color": "#E8F3F1"}),
    fac    = book.add_format({"num_format": "0.000", "border": 1}),
    pct    = book.add_format({"num_format": "+0.0%;-0.0%", "border": 1}),
    inlbl  = book.add_format({"bold": True, "bg_color": "#F1F5F8", "border": 1}),
    input  = book.add_format({"bg_color": "#FFF6E0", "border": 1, "locked": 0}),
    big    = book.add_format({"num_format": "$#,##0.00", "bold": True, "font_size": 14,
                              "font_color": "#0F2A43", "bg_color": "#E9C46A", "border": 1}),
)

# ===== Hoja FACTORES (el manual de tarifa) — con nombres de rango para los VLOOKUP =====
wsf = book.add_worksheet("Factores")
wsf.set_column("A:A", 20); wsf.set_column("B:B", 12)
wsf.write("A1", "Manual de tarifa — factores multiplicativos", F["titulo"])
wsf.write("A2", f"Prima base = ${PRIMA_BASE:,.2f}  ·  prima = base × Π factores", F["sub"])
RANGOS = {}          # var -> (rango_tabla, rango_niveles) para VLOOKUP / validación
r = 3
for v in VARIABLES:
    wsf.write(r, 0, NOMBRE[v], F["head"]); wsf.write(r, 1, "Factor", F["head"])
    datos_ini = r + 1
    for niv, fac in FACTORES[v]:
        wsf.write_string(r + 1, 0, niv, F["txt"]); wsf.write_number(r + 1, 1, fac, F["fac"])
        r += 1
    datos_fin = r
    RANGOS[v] = (f"Factores!{rng_abs(datos_ini,0,datos_fin,1)}",   # tabla (nivel,factor)
                 f"Factores!{rng_abs(datos_ini,0,datos_fin,0)}")   # solo niveles (dropdown)
    r += 2
wsf.protect()        # el manual de tarifa no se toca (hoja protegida)

# ===== Hoja COTIZADOR (vivo: dropdowns + fórmula, funciona sin Python) =====
wsc = book.add_worksheet("Cotizador"); book.worksheets_objs.insert(0, book.worksheets_objs.pop())
wsc.set_column("A:A", 20); wsc.set_column("B:B", 20); wsc.set_column("C:C", 12)
wsc.write("A1", "Cotizador de prima (Excel)", F["titulo"])
wsc.write("A2", "Elige el perfil en los menús; la prima se calcula sola.", F["sub"])
wsc.write("A4", "Prima base", F["inlbl"]); wsc.write_number("B4", PRIMA_BASE, F["money"])
wsc.write(4, 0, "Característica", F["head"]); wsc.write(4, 1, "Selección", F["head"])
wsc.write(4, 2, "Factor", F["head"])
fila0 = 5
for i, v in enumerate(VARIABLES):
    rr = fila0 + i
    wsc.write(rr, 0, NOMBRE[v], F["inlbl"])
    wsc.write_string(rr, 1, FACTORES[v][0][0], F["input"])                 # valor inicial
    wsc.data_validation(rr, 1, rr, 1, {"validate": "list", "source": "=" + RANGOS[v][1]})
    wsc.write_formula(rr, 2, f'=VLOOKUP({cell(rr,1)},{RANGOS[v][0]},2,FALSE)', F["fac"])
filaP = fila0 + len(VARIABLES) + 1
wsc.write(filaP, 0, "PRIMA PURA ESTIMADA", F["inlbl"])
wsc.write_formula(filaP, 1,
                  f'=B4*PRODUCT({cell(fila0,2)}:{cell(fila0+len(VARIABLES)-1,2)})', F["big"])
wsc.protect()        # solo las celdas de selección (locked=0) quedan editables

# ===== Hoja CONTROL (reconciliación Python vs Excel) =====
muestra = datos.sample(8, random_state=1).reset_index(drop=True)
wsk = book.add_worksheet("Control")
for j, v in enumerate(VARIABLES):
    wsk.set_column(j, j, 16); wsk.write(0, j, NOMBRE[v], F["head"])
wsk.write(0, len(VARIABLES),   "Prima Python", F["head"])
wsk.write(0, len(VARIABLES)+1, "Prima Excel",  F["head"])
wsk.write(0, len(VARIABLES)+2, "Diferencia",   F["head"])
for i in range(len(muestra)):
    rr = i + 1
    for j, v in enumerate(VARIABLES):
        wsk.write_string(rr, j, str(muestra.loc[i, v]), F["txt"])
    wsk.write_number(rr, len(VARIABLES), prima_python(muestra.loc[i]), F["money"])
    # misma fórmula que el cotizador, pero leyendo las celdas de ESTA fila
    prod = "*".join(f'VLOOKUP({cell(rr,j)},{RANGOS[v][0]},2,FALSE)'
                    for j, v in enumerate(VARIABLES))
    cE = len(VARIABLES) + 1
    wsk.write_formula(rr, cE, f'={PRIMA_BASE}*{prod}', F["money"])
    wsk.write_formula(rr, cE+1, f'={cell(rr,cE)}-{cell(rr,len(VARIABLES))}', F["money"])
wsk.write(len(muestra)+2, 0, "La columna Diferencia debe ser ~0: el Excel reproduce al modelo.",
          F["sub"])

# ===== Hoja REPRECIO (cartera repreciada + impacto vs prima base plana) =====
rep = datos.sample(200, random_state=2).reset_index(drop=True).copy()
rep["prima_glm"] = [prima_python(rep.loc[i]) for i in range(len(rep))]
rep["impacto"]   = rep["prima_glm"] / PRIMA_BASE - 1          # vs cobrar la base plana
cols = ["cobertura", "sexo", "combustible", "edad_conductor_cat",
        "potencia_cat", "nivel_bonus_cat", "prima_glm", "impacto"]
rep[cols].to_excel(wb, sheet_name="Reprecio", startrow=1, index=False, header=False)
wsr = wb.sheets["Reprecio"]
for j, c in enumerate(cols):
    wsr.write(0, j, c, F["head"])
    wsr.set_column(j, j, 15)
wsr.set_column(6, 6, 14); wsr.set_column(7, 7, 12)
# formato de las columnas numéricas + formato condicional en impacto
n = len(rep)
wsr.conditional_format(1, 7, n, 7, {"type": "cell", "criteria": ">", "value": 0,
                                    "format": book.add_format({"font_color": "#C44E52"})})
wsr.conditional_format(1, 7, n, 7, {"type": "cell", "criteria": "<=", "value": 0,
                                    "format": book.add_format({"font_color": "#2A9D8F"})})
# reaplicar num_format a las dos últimas columnas
mfmt = book.add_format({"num_format": "$#,##0.00"}); pfmt = book.add_format({"num_format": "+0.0%;-0.0%"})
for i in range(n):
    wsr.write_number(i+1, 6, rep["prima_glm"].iloc[i], mfmt)
    wsr.write_number(i+1, 7, rep["impacto"].iloc[i], pfmt)

# ===== Hoja RESUMEN/PORTADA (con una gráfica de factores de cobertura) =====
wsm = book.add_worksheet("Resumen"); book.worksheets_objs.insert(0, book.worksheets_objs.pop())
wsm.set_column("A:A", 24); wsm.set_column("B:B", 16); wsm.hide_gridlines(2)
wsm.write("A1", "Tarifa de auto — entregable", F["titulo"])
wsm.write("A2", "Generado desde Python (GLM Tweedie) · Diplomado ML Seguros", F["sub"])
wsm.write("A4", "Prima base", F["inlbl"]); wsm.write_number("B4", PRIMA_BASE, F["money_b"])
wsm.write("A5", "Pólizas en cartera", F["inlbl"]); wsm.write_number("B5", len(datos),
          book.add_format({"num_format": "#,##0", "border": 1}))
wsm.write("A6", "Hojas", F["inlbl"])
wsm.write("B6", "Factores · Cotizador · Control · Reprecio", F["txt"])
# gráfica: factores de cobertura (desde la hoja Factores)
rtab, _ = RANGOS["cobertura"]
chart = book.add_chart({"type": "column"})
a = rtab.split("!")[1].replace("$","").split(":")  # p.ej. A5:B7
import re
r0 = int(re.findall(r"\d+", a[0])[0]); r1 = int(re.findall(r"\d+", a[1])[0])
chart.add_series({"name": "Factor",
                  "categories": f"=Factores!$A${r0}:$A${r1}",
                  "values":     f"=Factores!$B${r0}:$B${r1}"})
chart.set_title({"name": "Factores de Cobertura"}); chart.set_legend({"none": True})
wsm.insert_chart("A8", chart, {"x_scale": 1.1, "y_scale": 1.1})

wb.close()
print("OK -> tarifa_cotizador.xlsx")
print(f"Prima base = ${PRIMA_BASE:,.2f}")
print("Reconciliación (primeros perfiles):")
for i in range(3):
    print(f"  perfil {i}: Python = ${prima_python(muestra.loc[i]):,.2f}")
