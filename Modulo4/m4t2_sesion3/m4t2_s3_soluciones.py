"""
m4t2_s3_soluciones.py
=====================
Soluciones a los ejercicios de la Sesión 3 — Módulo 4, Tema 2 (GLM con Python).
Diplomado ML en Seguros · FC UNAM · 28 de agosto de 2026

Prerequisito: haber corrido la Sesión 2 (existe modelos/metadatos.json) y cargar_modelos.py.
Uso:  python m4t2_s3_soluciones.py
"""
import os
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf
from cargar_modelos import cargar_modelos, preparar_datos

pd.set_option('display.float_format','{:,.4f}'.format); pd.set_option('display.width',120)

REF = {'cobertura':'Amplia','sexo':'Hombre','uso':'Particular','combustible':'Diesel',
       'edad_conductor_cat':'(35, 45]','antiguedad_vehiculo_cat':'(4, 5]',
       'potencia_cat':'(50, 60]','nivel_bonus_cat':'(-1, 0]'}


def _cargar():
    datos, sev = preparar_datos(); M = cargar_modelos()
    return datos, sev, M


def _ordenar(cats):
    return sorted(cats, key=lambda c: float(str(c).split(',')[0].strip('([')) if ',' in str(c) else str(c))


def efecto_parcial(datos, modelo, variable, con_offset=True):
    base = datos.iloc[[0]].copy()
    for k,v in REF.items(): base[k]=v
    base['exposicion']=1.0
    cats=_ordenar(datos[variable].astype(str).unique())
    grid=pd.concat([base.assign(**{variable:c}) for c in cats], ignore_index=True)
    pred = modelo.predict(grid, offset=np.log(grid['exposicion'])) if con_offset else modelo.predict(grid)
    return pd.DataFrame({variable:cats, 'efecto':pred.values})


def gini(y, yhat):
    def g(orden):
        s=np.asarray(y,float)[np.argsort(orden)]; n=len(s); tot=s.sum()
        return (2*np.sum(np.cumsum(s))-(n+1)*tot)/(n*tot)
    return g(yhat)/g(y)


# ── EJERCICIO 1 — efecto parcial ─────────────────────────────────────────────
def solucion_ejercicio1():
    print("="*65); print("  EJERCICIO 1 — Efecto parcial de la antigüedad"); print("="*65)
    datos, sev, M = _cargar()
    ef = efecto_parcial(datos, M['frecuencia'], 'antiguedad_vehiculo_cat')
    print("\n1a. Efecto parcial (frecuencia):")
    print(ef.round(4).to_string(index=False))
    tend = 'monótona' if (np.diff(ef['efecto'])>0).all() or (np.diff(ef['efecto'])<0).all() else 'no monótona (con mínimo/máximo intermedio)'
    print(f"\n1b. La forma es {tend}. Suele tener sentido: autos muy nuevos y muy viejos")
    print("    difieren en riesgo; a veces hay un mínimo en antigüedad intermedia.")
    ef_sev = efecto_parcial(datos, M['severidad'], 'antiguedad_vehiculo_cat', con_offset=False)
    rango_f = ef['efecto'].max()/ef['efecto'].min()
    rango_s = ef_sev['efecto'].max()/ef_sev['efecto'].min()
    print(f"\n1c. Rango del efecto — frecuencia: {rango_f:.2f}x  ·  severidad: {rango_s:.2f}x")
    print(f"    La severidad suele ser {'MÁS plana' if rango_s<rango_f else 'menos plana'} que la frecuencia.")
    print()


# ── EJERCICIO 2 — validación ─────────────────────────────────────────────────
def solucion_ejercicio2():
    print("="*65); print("  EJERCICIO 2 — Interpreta la validación"); print("="*65)
    datos, sev, M = _cargar()
    np.random.seed(42); idx=np.random.rand(len(datos))<0.8
    tr,te=datos[idx].copy(),datos[~idx].copy()
    m=smf.glm(M['meta']['frecuencia']['formula'],tr,family=sm.families.Poisson(),offset=np.log(tr.exposicion)).fit()
    te['pred']=m.predict(te,offset=np.log(te.exposicion)); te['pred_freq']=te['pred']/te['exposicion']
    ratio=te['pred'].sum()/te['num_siniestros'].sum(); g=gini(te.num_siniestros.values, te.pred.values)
    print(f"\n2a. Ratio pred/obs = {ratio:.4f}. Cercano a 1 → el modelo está bien calibrado")
    print("    globalmente: predice casi el total de siniestros observados en test.")
    print(f"\n2b. Gini = {g:.4f}. {'Supera' if g>0.30 else 'No supera'} 0.30, así que discrimina de forma")
    print(f"    {'aceptable' if g>0.30 else 'modesta'}: separa {'bien' if g>0.30 else 'de forma limitada'} riesgos altos de bajos.")
    te['decil']=pd.qcut(te['pred_freq'],10,labels=False,duplicates='drop')
    cal=te.groupby('decil').apply(lambda x: pd.Series({'obs':x.num_siniestros.sum()/x.exposicion.sum(),
        'pred':x.pred.sum()/x.exposicion.sum()}),include_groups=False)
    cal['dif_rel']=(cal['pred']/cal['obs']-1).abs()
    peor=cal['dif_rel'].idxmax()
    print(f"\n2c. El decil peor calibrado es el {int(peor)+1} (dif rel {cal.loc[peor,'dif_rel']:.1%}).")
    print("    Si un decil se desvía mucho, revisar si falta una variable o un banding en ese segmento.")
    print()


# ── EJERCICIO 3 — deducible ──────────────────────────────────────────────────
def solucion_ejercicio3():
    print("="*65); print("  EJERCICIO 3 — El efecto del deducible"); print("="*65)
    datos, sev, M = _cargar()
    con=datos[datos.num_siniestros>0]; expo=datos.exposicion.sum()
    freq0=datos.num_siniestros.sum()/expo
    print(f"\n   Frecuencia sin deducible: {freq0:.4f}")
    print("\n3a. Frecuencia y severidad con deducibles $750 y $1,500:")
    for D in [750, 1500]:
        rep=con[con.monto_total>=D]
        f=rep.num_siniestros.sum()/expo
        sv=np.average(rep.severidad, weights=rep.num_siniestros)
        print(f"    Deducible ${D:>5}: freq {f:.4f} (cae {(1-f/freq0):.0%})  ·  sev media ${sv:,.0f}")
    print("\n3b. La severidad media SUBE porque al quitar los siniestros pequeños (< deducible)")
    print("    quedan solo los grandes: cambia la MEZCLA, no el tamaño de cada uno.")
    print("\n3c. El deducible no debe entrar tal cual en severidad porque su efecto es de")
    print("    SELECCIÓN (qué siniestros se observan), no de tamaño del siniestro; meterlo")
    print("    ahí confundiría el efecto de no-reporte con el efecto real sobre el monto.")
    print()


# ── EJERCICIO 4 — sensibilidad ───────────────────────────────────────────────
def solucion_ejercicio4():
    print("="*65); print("  EJERCICIO 4 — Sensibilidad de otro factor"); print("="*65)
    datos, sev, M = _cargar()
    datos['pp'] = (M['frecuencia'].predict(datos, offset=np.log(datos.exposicion))/datos.exposicion) * M['severidad'].predict(datos)
    base = np.average(datos['pp'], weights=datos['exposicion'])

    def impacto(col, nivel, chg):
        d=datos.copy(); mask=d[col].astype(str)==nivel
        d.loc[mask,'pp']*=(1+chg/100)
        return (np.average(d['pp'],weights=d['exposicion'])/base-1)*100, datos.loc[mask,'exposicion'].sum()/datos.exposicion.sum()

    niveles = _ordenar(datos['nivel_bonus_cat'].astype(str).unique())
    alto = niveles[-1]
    imp_alto, expo_alto = impacto('nivel_bonus_cat', alto, 15)
    # un nivel de baja exposición
    expos = datos.groupby('nivel_bonus_cat', observed=True)['exposicion'].sum()
    bajo = str(expos.idxmin())
    imp_bajo, expo_bajo = impacto('nivel_bonus_cat', bajo, 15)
    print(f"\n4a. +15% a nivel_bonus {alto} (expo {expo_alto:.1%}): impacto {imp_alto:+.2f}% en prima total")
    print(f"\n4b. +15% a nivel_bonus {bajo} (expo {expo_bajo:.1%}): impacto {imp_bajo:+.2f}% en prima total")
    print(f"    Mueve más la prima total el nivel con MAYOR exposición.")
    print("\n4c. Conviene afinar la tarifa donde hay masa de cartera: ahí un ajuste pequeño")
    print("    tiene impacto agregado real; en segmentos chicos, casi no mueve el total.")
    print()


# ── INTEGRADOR — simulación de tarifa ────────────────────────────────────────
def solucion_integrador():
    print("="*65); print("  INTEGRADOR — Simulación de tarifa"); print("="*65)
    datos, sev, M = _cargar()
    # 1) prima pura
    datos['pp'] = (M['frecuencia'].predict(datos, offset=np.log(datos.exposicion))/datos.exposicion) * M['severidad'].predict(datos)
    pp_total = (datos['pp']*datos['exposicion']).sum()
    # 2) prima de tarifa
    GASTOS, MARGEN = 0.25, 0.05; recargo = 1/(1-GASTOS-MARGEN)
    datos['prima_tarifa'] = datos['pp']*recargo
    pt_total = (datos['prima_tarifa']*datos['exposicion']).sum()
    print(f"\n[1-2] Prima pura total ${pp_total:,.0f} · recargo ×{recargo:.3f} · prima tarifa total ${pt_total:,.0f}")
    # 3) escenario
    d=datos.copy()
    joven=d['edad_conductor_cat'].astype(str)=='(17, 30]'
    d.loc[joven,'prima_tarifa']*=1.08
    nueva=(d['prima_tarifa']*d['exposicion']).sum()
    print(f"[3] Escenario (+8% jóvenes): prima tarifa ${nueva:,.0f} ({(nueva/pt_total-1)*100:+.2f}%)")
    # 4) cobertura de la prima pura
    print(f"[4] ¿Cubre la prima pura? Prima tarifa nueva ${nueva:,.0f} >= prima pura ${pp_total:,.0f}: "
          f"{'SÍ' if nueva>=pp_total else 'NO'}")
    # 5) documentación
    print("\n[5] Nota técnica (ejemplo):")
    print("    'La tarifa se construye como Frecuencia × Severidad, cargando 25% de gastos y 5% de")
    print("     margen (recargo ×1.429). El escenario propuesto recarga 8% al segmento de conductores")
    print("     jóvenes (17-30), el de mayor prima pura, elevando la recaudación total 2.5% y")
    print("     manteniendo cobertura sobre la prima pura del portafolio.'")
    print()


if __name__ == '__main__':
    print("\nSESIÓN 3 — SOLUCIONES COMPLETAS")
    print("Diplomado ML en Seguros · FC UNAM · 28 de agosto 2026"); print("="*65)
    if not os.path.exists('modelos/metadatos.json'):
        print("ERROR: falta modelos/metadatos.json. Corre primero la Sesión 2."); raise SystemExit(1)
    solucion_ejercicio1(); solucion_ejercicio2(); solucion_ejercicio3()
    solucion_ejercicio4(); solucion_integrador()
    print("Todas las soluciones se ejecutaron correctamente.")
