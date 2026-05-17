"""
sesion8_soluciones.py
=====================
Soluciones completas a los 3 ejercicios intermedios
y al ejercicio integrador final de la Sesion 8.

Prerequisito: haber ejecutado generar_datos.py primero.

Uso:
    python sesion8_soluciones.py

O ejecutar cada seccion por separado en el notebook
descomentando el bloque correspondiente.
"""

from zipfile import Path

import pandas as pd
import numpy as np
import os, time, json

pd.set_option('display.float_format', '{:,.2f}'.format)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 120)

# ─────────────────────────────────────────────────────────────────────────────
# COLUMNAS PARA CARGAR — definidas una sola vez
# ─────────────────────────────────────────────────────────────────────────────
ANALITICAS = [
    'id_poliza','num_poliza','ramo','plan','status_poliza',
    'nombre','apellido_paterno','apellido_materno',
    'rfc','edad','sexo','estado_civil','ocupacion',
    'fecha_emision','fecha_inicio_vigencia','fecha_fin_vigencia',
    'num_renovaciones','motivo_baja',
    'suma_asegurada','deducible','prima_neta','prima_total',
    'forma_pago','agente_id','canal_venta',
    'estado','municipio','codigo_postal',
    'marca_vehiculo','modelo_vehiculo','tipo_vehiculo',
]

MAPA_SEXO = {
    'M':'M','MASCULINO':'M','HOMBRE':'M','MASC':'M',
    'F':'F','FEMENINO':'F','MUJER':'F','FEM':'F',
}


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES DE LIMPIEZA REUTILIZABLES
# ─────────────────────────────────────────────────────────────────────────────
def cargar_cartera(ruta='datos/cartera_polizas.csv', cols=None):
    """Carga la cartera con las columnas correctas y na_values."""
    return pd.read_csv(
        ruta,
        usecols=cols or ANALITICAS,
        na_values=['N/D','N/A','ND','--','Sin dato',''],
    )


def limpiar_fechas(df):
    """Convierte todas las columnas de fecha a datetime."""
    # Fecha nacimiento: formato d/m/Y (sistema legacy)
    if 'fecha_nacimiento' in df.columns:
        df['fecha_nacimiento'] = pd.to_datetime(
            df['fecha_nacimiento'], format='%d/%m/%Y', errors='coerce')

    # Fechas ISO estandar
    for col in ['fecha_emision','fecha_inicio_vigencia','fecha_fin_vigencia']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Fechas en formato d/m/Y en siniestros
    for col in ['fecha_apertura','fecha_ultimo_movimiento']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format='%d/%m/%Y', errors='coerce')

    # Fechas ISO en siniestros
    for col in ['fecha_ocurrencia','fecha_reporte','fecha_cierre']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    return df


def normalizar_texto(df):
    """Normaliza campos de texto sucios."""
    if 'sexo' in df.columns:
        df['sexo'] = df['sexo'].str.strip().str.upper().map(MAPA_SEXO)
    if 'codigo_postal' in df.columns:
        df['codigo_postal'] = df['codigo_postal'].replace('N/D', np.nan)
    return df


def optimizar_tipos(df, umbral_cat=0.05, verbose=True):
    """
    Optimiza tipos de dato de forma segura.
    - Categoricas: columnas texto con < umbral_cat% de valores unicos
    - Enteros: downcast seguro segun rango
    - Floats de montos: SE CONSERVAN en float64
    """
    df = df.copy()
    mb_antes = df.memory_usage(deep=True).sum() / 1024**2

    # Categoricas
    for col in df.select_dtypes('object').columns:
        pct = df[col].nunique() / len(df)
        if pct < umbral_cat:
            df[col] = df[col].astype('category')

    # Enteros: downcast seguro
    for col in df.select_dtypes('int64').columns:
        mx = df[col].abs().max()
        if mx <= 127:    df[col] = df[col].astype('int8')
        elif mx <= 32767:df[col] = df[col].astype('int16')
        elif mx <= 2_147_483_647: df[col] = df[col].astype('int32')

    # Columnas de monto: NO hacer downcast a float32
    # (pueden tener valores > 100k con centavos importantes)

    mb_desp = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        reduccion = (1 - mb_desp/mb_antes)*100
        print(f"  Memoria: {mb_antes:.1f} MB → {mb_desp:.1f} MB ({reduccion:.0f}% reduccion)")
    return df


def enriquecer_cartera(df, ramos_df, agentes_df):
    """Agrega columnas derivadas y hace los merges con catalogos."""
    hoy = pd.Timestamp.today()

    # Merges con catalogos
    df = pd.merge(df, ramos_df[['ramo','nombre_largo','tasa_base']],
                  on='ramo', how='left')
    df = pd.merge(df, agentes_df[['agente_id','nombre','region','tipo']].rename(
                  columns={'nombre':'nombre_agente','tipo':'tipo_agente'}),
                  on='agente_id', how='left')

    # Grupo de edad
    df['g_edad'] = pd.cut(
        df['edad'], bins=[0,30,45,60,100],
        labels=['18-30','31-45','46-60','61+'])

    # Prima calculada por tarifa
    df['prima_calc'] = (df['suma_asegurada'] * df['tasa_base'] * 1.16).round(2)

    # Diferencia prima real vs calculada
    df['diferencia_prima'] = (df['prima_total'] - df['prima_calc']).round(2)

    # Nivel de riesgo por prima (sustituye mi_modulo si no esta disponible)
    df['nivel_riesgo'] = pd.cut(
        df['prima_total'],
        bins=[0, 4000, 10000, float('inf')],
        labels=['BAJO','MEDIO','ALTO'])

    # Calculos de vigencia
    if 'fecha_inicio_vigencia' in df.columns and pd.api.types.is_datetime64_any_dtype(df['fecha_inicio_vigencia']):
        df['dias_vigencia']     = (df['fecha_fin_vigencia'] - df['fecha_inicio_vigencia']).dt.days
        dias_trans              = (hoy - df['fecha_inicio_vigencia']).dt.days
        df['fraccion_expuesta'] = (dias_trans / df['dias_vigencia']).clip(0, 1).round(4)

    # Componentes de fecha para agrupacion temporal
    if 'fecha_emision' in df.columns and pd.api.types.is_datetime64_any_dtype(df['fecha_emision']):
        df['anio_emision']      = df['fecha_emision'].dt.year
        df['trimestre_emision'] = df['fecha_emision'].dt.quarter

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SOLUCION EJERCICIO 1 — Auditoria de columnas
# ─────────────────────────────────────────────────────────────────────────────
def solucion_ejercicio1():
    print("=" * 65)
    print("  SOLUCION EJERCICIO 1 — Auditoria de columnas")
    print("=" * 65)

    df = cargar_cartera()

    # 1a: columnas con mas NaN
    print("\n1a. Las 5 columnas con mas NaN:")
    nan_pct = (df.isna().mean() * 100).sort_values(ascending=False)
    for col, pct in nan_pct.head(5).items():
        sentido = {
            'deducible':   '✓ Esperado — polizas de Vida no tienen deducible',
            'motivo_baja': '✓ Esperado — solo polizas canceladas tienen motivo',
            'marca_vehiculo': '✓ Esperado — solo Autos tiene vehiculo',
            'modelo_vehiculo':'✓ Esperado — solo Autos tiene vehiculo',
            'tipo_vehiculo':  '✓ Esperado — solo Autos tiene vehiculo',
            'ocupacion':   '? Posible error — 8% de asegurados sin ocupacion registrada',
            'prima_neta':  '✗ Error de captura — primas faltantes requieren imputacion',
        }.get(col, '? Revisar con el area de sistemas')
        print(f"  {col:<25}: {pct:>5.1f}% NaN  → {sentido}")

    # 1b: columnas candidatas a category
    print("\n1b. Columnas con < 5% valores unicos → candidatas a 'category':")
    for col in df.select_dtypes('object').columns:
        n_uniq = df[col].nunique()
        pct    = n_uniq / len(df) * 100
        if pct < 5:
            print(f"  {col:<25}: {n_uniq:>5} valores unicos ({pct:.2f}%)")

    # 1c: rango imposible
    print("\n1c. Diagnostico de rangos en columnas numericas:")
    print(df[['edad','suma_asegurada','prima_neta','prima_total','num_renovaciones']].describe().round(2))
    print("\n  Nota: num_renovaciones con 0-4 es correcto.")
    print("  Verificar si prima_neta tiene valores negativos (error de captura):")
    negativos = df[df['prima_neta'] < 0]['prima_neta'] if 'prima_neta' in df.columns else pd.Series()
    print(f"  Primas negativas: {len(negativos)} (si hay, son errores de captura)")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# SOLUCION EJERCICIO 2 — Limpiar siniestros.csv
# ─────────────────────────────────────────────────────────────────────────────
def solucion_ejercicio2():
    print("=" * 65)
    print("  SOLUCION EJERCICIO 2 — Limpiar siniestros.csv")
    print("=" * 65)

    COLS_SINIESTROS = [
        'id_siniestro','id_poliza','ramo','tipo_siniestro',
        'fecha_ocurrencia','fecha_reporte','fecha_ultimo_movimiento','fecha_cierre',
        'dias_reporte','monto_reclamado','monto_pagado',
        'status_siniestro','motivo_rechazo','id_ajustador',
    ]

    # 2a: carga selectiva
    df_sin = pd.read_csv(
        'datos/siniestros.csv',
        usecols=COLS_SINIESTROS,
        na_values=['N/A','','ND'],
    )
    print(f"\n2a. Cargado: {df_sin.shape}  (descartamos 16 columnas administrativas/redundantes)")

    # 2b: convertir fechas
    df_sin['fecha_ocurrencia'] = pd.to_datetime(df_sin['fecha_ocurrencia'], errors='coerce')
    df_sin['fecha_reporte']    = pd.to_datetime(df_sin['fecha_reporte'],    errors='coerce')
    df_sin['fecha_cierre']     = pd.to_datetime(df_sin['fecha_cierre'],     errors='coerce')
    df_sin['fecha_ultimo_movimiento'] = pd.to_datetime(
        df_sin['fecha_ultimo_movimiento'], format='%d/%m/%Y', errors='coerce')

    n_nat = df_sin['fecha_ultimo_movimiento'].isna().sum()
    print(f"2b. Fechas convertidas. NaT en fecha_ultimo_movimiento: {n_nat}")

    # 2c: dias de reporte real
    df_sin['dias_reporte_real'] = (
        df_sin['fecha_reporte'] - df_sin['fecha_ocurrencia']).dt.days

    # 2d: dias de resolucion real
    df_sin['dias_resolucion_real'] = (
        df_sin['fecha_cierre'] - df_sin['fecha_reporte']).dt.days
    # NaT si no esta cerrado → queda como NaN automaticamente

    # 2e: mas de 180 dias sin cerrar
    hoy = pd.Timestamp.today()
    abiertos = df_sin[df_sin['fecha_cierre'].isna()].copy()
    abiertos['dias_abierto'] = (hoy - abiertos['fecha_reporte']).dt.days
    criticos = abiertos[abiertos['dias_abierto'] > 180]

    print(f"\n2e. Siniestros abiertos:              {len(abiertos):,}")
    print(f"    Con mas de 180 dias sin cerrar:   {len(criticos):,}")
    print(f"    Distribucion por ramo:")
    print(criticos['ramo'].value_counts().to_string())

    print(f"\n    Muestra de siniestros criticos:")
    cols_show = ['id_siniestro','ramo','tipo_siniestro','dias_abierto','monto_reclamado']
    print(criticos[cols_show].head(5).to_string(index=False))

    print()
    return df_sin


# ─────────────────────────────────────────────────────────────────────────────
# SOLUCION EJERCICIO 3 — Funcion limpiar_cartera()
# ─────────────────────────────────────────────────────────────────────────────
def limpiar_cartera_completa(name_csv = 'cartera_polizas.csv'):
    """
    Pipeline completo de limpieza de la cartera.

    Pasos:
    1. Carga selectiva (solo columnas analiticas)
    2. Elimina duplicados
    3. Normaliza sexo
    4. Convierte fechas
    5. Imputa prima_neta por mediana de ramo
    6. Limpia codigo_postal
    7. Optimiza tipos de dato

    Retorna: DataFrame limpio y optimizado
    """
    
    from pathlib import Path
    RUTA_BASE = Path(__file__).parent   # carpeta donde vive el .py
    ruta_csv = RUTA_BASE / 'datos' / name_csv

    print("  [1/7] Cargando columnas analiticas...")
    mb_csv = os.path.getsize(ruta_csv) / 1024**2
    df = cargar_cartera(ruta_csv)
    mb_cargado = df.memory_usage(deep=True).sum() / 1024**2
    print(f"        CSV: {mb_csv:.1f} MB → en memoria: {mb_cargado:.1f} MB · Shape: {df.shape}")

    print("  [2/7] Eliminando duplicados...")
    n_antes = len(df)
    df = df.drop_duplicates()
    print(f"        Eliminados: {n_antes - len(df)} duplicados · Quedan: {len(df):,}")

    print("  [3/7] Normalizando texto (sexo, codigo_postal)...")
    df = normalizar_texto(df)
    sexo_nans = df['sexo'].isna().sum()
    print(f"        Sexo normalizado. Valores no reconocidos → NaN: {sexo_nans}")

    print("  [4/7] Convirtiendo fechas...")
    df = limpiar_fechas(df)
    nat_total = sum(df[c].isna().sum()
                    for c in df.columns
                    if pd.api.types.is_datetime64_any_dtype(df[c]))
    print(f"        NaT generados por fechas invalidas: {nat_total}")

    print("  [5/7] Imputando prima_neta por mediana de ramo...")
    n_nan_prima = df['prima_neta'].isna().sum()
    df['prima_neta'] = df.groupby('ramo')['prima_neta'].transform(
        lambda x: x.fillna(x.median()))
    print(f"        Imputadas: {n_nan_prima} primas → mediana de su ramo")

    print("  [6/7] Limpiando campos adicionales...")
    df['codigo_postal'] = df['codigo_postal'].replace('N/D', np.nan)
    cp_nan = df['codigo_postal'].isna().sum()
    print(f"        CPs invalidos convertidos a NaN: {cp_nan}")

    print("  [7/7] Optimizando tipos de dato...")
    df = optimizar_tipos(df, verbose=True)

    # Verificacion final
    total_nan = df.isna().sum().sum()
    print(f"\n  ✓ Verificacion final:")
    print(f"    Shape:           {df.shape}")
    print(f"    NaN restantes:   {total_nan} (normales: deducible Vida, vehiculo no-Autos, etc.)")
    print(f"    Duplicados:      {df.duplicated().sum()}")
    print(f"    Sexo sucio:      {df['sexo'].isin(['MASCULINO','FEMENINO','m','f']).sum()}")
    print(f"    Fechas como str: {df.select_dtypes('object').filter(like='fecha').shape[1]}")
    return df


def solucion_ejercicio3():
    print("=" * 65)
    print("  SOLUCION EJERCICIO 3 — Pipeline limpiar_cartera()")
    print("=" * 65)
    print()
    df = limpiar_cartera_completa()
    print()
    print("  Primeras 3 filas del dataset limpio:")
    print(df[['id_poliza','ramo','sexo','fecha_emision','prima_neta']].head(3).to_string(index=False))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SOLUCION EJERCICIO INTEGRADOR FINAL
# ─────────────────────────────────────────────────────────────────────────────
def solucion_integrador():
    print("=" * 65)
    print("  SOLUCION EJERCICIO INTEGRADOR — Pipeline Completo")
    print("=" * 65)
    os.makedirs('datos', exist_ok=True)

    # ── FASE 1: Ingesta inteligente ───────────────────────────────────────────
    print("\n[FASE 1] Ingesta inteligente")
    t0 = time.time()

    df      = cargar_cartera()
    ramos   = pd.read_csv('datos/catalogo_ramos.csv')
    agentes = pd.read_csv('datos/catalogo_agentes.csv')

    mb_selectivo = df.memory_usage(deep=True).sum() / 1024**2
    # Para comparar: cuanto pesaria cargar todo
    df_full = pd.read_csv('datos/cartera_polizas.csv')
    mb_full = df_full.memory_usage(deep=True).sum() / 1024**2
    del df_full

    print(f"  Carga completa (46 cols):  {mb_full:.1f} MB")
    print(f"  Carga selectiva ({len(ANALITICAS)} cols): {mb_selectivo:.1f} MB  ({(1-mb_selectivo/mb_full)*100:.0f}% menos)")

    # ── FASE 2: Limpieza completa ─────────────────────────────────────────────
    print("\n[FASE 2] Limpieza completa")

    # Duplicados
    n_antes = len(df)
    df = df.drop_duplicates()
    print(f"  Duplicados eliminados: {n_antes - len(df)}")

    # Normalizar sexo
    df = normalizar_texto(df)

    # Convertir fechas
    df = limpiar_fechas(df)

    # Imputar prima_neta por mediana de ramo (no mediana global)
    # Decision: mediana de ramo porque GMM tiene prima ~3x mayor que Autos
    n_imp = df['prima_neta'].isna().sum()
    df['prima_neta'] = df.groupby('ramo')['prima_neta'].transform(
        lambda x: x.fillna(x.median()))
    print(f"  prima_neta imputada (mediana por ramo): {n_imp} valores")

    # Optimizar tipos
    df = optimizar_tipos(df, verbose=True)

    # ── FASE 3: Enriquecimiento ───────────────────────────────────────────────
    print("\n[FASE 3] Enriquecimiento")
    df = enriquecer_cartera(df, ramos, agentes)
    print(f"  Shape despues de enriquecimiento: {df.shape}")
    print(f"  Columnas nuevas: nombre_largo, tasa_base, nombre_agente, g_edad, prima_calc, nivel_riesgo")

    # ── FASE 4: Analisis y reportes ───────────────────────────────────────────
    print("\n[FASE 4] Analisis multidimensional")

    # Resumen por ramo
    resumen_ramo = df.groupby('nombre_largo').agg(
        polizas       = ('id_poliza',   'count'),
        prima_total   = ('prima_total', 'sum'),
        prima_prom    = ('prima_total', 'mean'),
        prima_calc_tot= ('prima_calc',  'sum'),
        siniest_proxy = ('nivel_riesgo', lambda x: (x=='ALTO').sum()),
    ).round(2).reset_index()
    resumen_ramo['pct_cartera'] = (
        resumen_ramo['prima_total'] / resumen_ramo['prima_total'].sum() * 100
    ).round(1)
    resumen_ramo['frec_alto_riesgo'] = (
        resumen_ramo['siniest_proxy'] / resumen_ramo['polizas'] * 100
    ).round(2)

    print("\n  Resumen por ramo:")
    print(resumen_ramo[['nombre_largo','polizas','prima_total','pct_cartera','frec_alto_riesgo']].to_string(index=False))

    # Resumen por agente (top 10)
    resumen_agente = df.groupby('nombre_agente').agg(
        polizas    = ('id_poliza',    'count'),
        prima_total= ('prima_total',  'sum'),
    ).round(2).reset_index().sort_values('prima_total', ascending=False)
    resumen_agente['comision_est'] = (resumen_agente['prima_total'] * 0.10).round(2)

    print(f"\n  Top 5 agentes por prima:")
    print(resumen_agente.head(5).to_string(index=False))

    # Pivot: prima por ramo × grupo de edad
    pivot_prima = pd.pivot_table(
        df, values='prima_total', index='nombre_largo',
        columns='g_edad', aggfunc='sum', fill_value=0, margins=True, margins_name='TOTAL'
    ).round(0) / 1000  # en miles MXN

    # Pivot: polizas por estado × ramo
    pivot_zona = pd.pivot_table(
        df, values='id_poliza', index='estado',
        columns='ramo', aggfunc='count', fill_value=0, margins=True, margins_name='TOTAL'
    )

    print(f"\n  Pivot prima por ramo y edad (miles MXN):")
    print(pivot_prima.to_string())

    # Hallazgo clave
    ramo_max = resumen_ramo.loc[resumen_ramo['prima_total'].idxmax(), 'nombre_largo']
    zona_max = (df.groupby('estado')['prima_total'].sum().idxmax())
    print(f"\n  Hallazgos:")
    print(f"  → Ramo con mayor prima total: {ramo_max}")
    print(f"  → Estado con mayor prima:     {zona_max}")

    # ── FASE 5: Exportar ──────────────────────────────────────────────────────
    print("\n[FASE 5] Exportar resultados")

    # Excel con 5 hojas
    ruta_xl = 'datos/reporte_ejecutivo_Q1_2026.xlsx'
    with pd.ExcelWriter(ruta_xl, engine='openpyxl') as writer:
        # Solo columnas analiticas + derivadas relevantes para el Excel
        cols_excel = ['id_poliza','ramo','nombre_largo','g_edad','nivel_riesgo',
                      'prima_neta','prima_total','prima_calc','fraccion_expuesta',
                      'estado','municipio','nombre_agente','canal_venta']
        cols_excel = [c for c in cols_excel if c in df.columns]
        df[cols_excel].to_excel(writer, sheet_name='Cartera_Limpia', index=False)
        resumen_ramo.to_excel(writer, sheet_name='Resumen_Ramo',   index=False)
        resumen_agente.to_excel(writer, sheet_name='Resumen_Agente', index=False)
        pivot_prima.to_excel(writer, sheet_name='Pivot_Prima')
        pivot_zona.to_excel(writer, sheet_name='Pivot_Zona')

    kb_xl = os.path.getsize(ruta_xl) / 1024
    print(f"  Excel generado: {ruta_xl}  ({kb_xl:.0f} KB · 5 hojas)")

    # Parquet optimizado
    ruta_pq = 'datos/cartera_q1_2026_final.parquet'
    df.to_parquet(ruta_pq, index=False)
    kb_pq = os.path.getsize(ruta_pq) / 1024

    # CSV equivalente para comparar
    ruta_csv_exp = 'datos/cartera_q1_2026_final_comparar.csv'
    df.to_csv(ruta_csv_exp, index=False)
    kb_csv = os.path.getsize(ruta_csv_exp) / 1024

    print(f"  Parquet: {kb_pq:.0f} KB")
    print(f"  CSV eq.: {kb_csv:.0f} KB  (Parquet es {kb_csv/kb_pq:.1f}x mas compacto)")
    os.remove(ruta_csv_exp)

    t_total = time.time() - t0
    print(f"\n  Pipeline completo en {t_total:.1f} segundos")
    print(f"  Dataset final: {df.shape[0]:,} polizas · {df.shape[1]} columnas")
    print()

    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — ejecutar todas las soluciones en orden
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\nSESOION 8 — SOLUCIONES COMPLETAS")
    print("Diplomado ML en Seguros · FC UNAM · 2 de mayo 2026")
    print("=" * 65)

    if not os.path.exists('datos/cartera_polizas.csv'):
        print("ERROR: ejecuta primero: python generar_datos.py")
        exit(1)

    solucion_ejercicio1()
    solucion_ejercicio2()
    solucion_ejercicio3()
    solucion_integrador()

    print("Todas las soluciones ejecutadas correctamente.")
    print("Revisa la carpeta datos/ para ver los archivos generados.")
