"""
soluciones_m2t2.py
==================
Soluciones a los ejercicios del Modulo 2, Tema 2.
Ejecuta: python soluciones_m2t2.py [e1|e2|final]
"""
import pandas as pd, numpy as np, matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import time, sys, warnings
warnings.filterwarnings('ignore')

import os
rutas = [
    'datos/tabla_maestra_s9.parquet',
    '../sesion_10/datos/cartera_q1_2026_final.parquet',
    'datos/cartera_q1_2026_final.parquet',
]
ruta = next((r for r in rutas if os.path.exists(r)), None)
if ruta is None:
    raise FileNotFoundError('No se encontro el parquet.')
df_raw = pd.read_parquet(ruta)
np.random.seed(2026)
N = len(df_raw)
idx_prima = np.random.choice(N, int(N*0.02), replace=False)
df_raw.loc[df_raw.index[idx_prima], 'prima_neta'] = np.nan
df_raw.loc[df_raw['ramo']=='Vida', 'deducible'] = np.nan
if 'ocupacion' in df_raw.columns:
    prob = ((df_raw['edad']-18)/(70-18))*0.20
    df_raw.loc[np.random.binomial(1,prob)==1, 'ocupacion'] = np.nan

features_num = ['prima_neta','suma_asegurada','edad','deducible']
df_gmm = df_raw[df_raw['ramo']=='GMM'][features_num].copy()
prima_original = df_gmm['prima_neta'].dropna()


def solucion_e1():
    print("="*60)
    print("  SOLUCION E1 — Comparar Estrategias de Imputacion")
    print("="*60)
    estrategias = {
        'Media':    SimpleImputer(strategy='mean'),
        'Mediana':  SimpleImputer(strategy='median'),
        'Moda':     SimpleImputer(strategy='most_frequent'),
        'Constante':SimpleImputer(strategy='constant', fill_value=0),
    }
    print(f"\nOriginal — media: ${prima_original.mean():,.2f}  "
          f"mediana: ${prima_original.median():,.2f}  "
          f"std: ${prima_original.std():,.2f}")
    print()
    resultados = []
    for nombre, imp in estrategias.items():
        df_imp = pd.DataFrame(imp.fit_transform(df_gmm), columns=features_num)
        serie  = df_imp['prima_neta']
        ks, pv = stats.ks_2samp(prima_original, serie)
        resultados.append(dict(
            estrategia=nombre,
            media=round(serie.mean(),2),
            mediana=round(serie.median(),2),
            std=round(serie.std(),2),
            ks=round(ks,4), p_value=round(pv,4),
            distorsiona='NO' if pv>0.05 else 'SI'
        ))
        print(f"{nombre:<12} media=${serie.mean():,.2f}  "
              f"mediana=${serie.median():,.2f}  "
              f"KS={ks:.4f}  p={pv:.4f}  {'OK' if pv>0.05 else 'DISTORSIONA'}")
    print()
    mejor = max(resultados, key=lambda x: x['p_value'])
    print(f"Recomendacion: {mejor['estrategia']} "
          f"(p={mejor['p_value']:.4f} — mayor similitud con original)")
    print("Para prima_neta con outliers, Mediana es la mas robusta.")


def solucion_e2():
    print("="*60)
    print("  SOLUCION E2 — KNN vs MICE — MAE Comparativo")
    print("="*60)

    # ── Preparar datos con patron MAR ────────────────────────────────────────────
    # IMPORTANTE: empezar con filas COMPLETAS (sin NaN previos del setup)
    # df_gmm puede tener NaN del 2% introducido en la celda de setup
    # Si no limpiamos, prima_verdadera_arr contiene NaN y MAE explota
    np.random.seed(42)
    df_demo = df_gmm.dropna().copy().reset_index(drop=True)  # solo filas completas

    prob_nan = (df_demo['suma_asegurada'] / df_demo['suma_asegurada'].max()) * 0.3
    mask_nan_bool = (np.random.binomial(1, prob_nan.fillna(0)) == 1)

    # Guardar los valores reales ANTES de poner NaN — ahora sin NaN garantizado
    prima_verdadera_arr = df_demo['prima_neta'].values.copy()

    df_demo.loc[mask_nan_bool, 'prima_neta'] = np.nan
    print(f"Filas de trabajo: {len(df_demo):,}  |  NaN introducidos: {mask_nan_bool.sum()}")

    # Normalizar para KNN y MICE — fit solo en filas completas
    scaler = StandardScaler()
    scaler.fit(df_demo.dropna())  # fit solo en filas completas
    X_sc = scaler.transform(df_demo)  # transform tolera NaN — los imputers los llenan  # contiene NaN donde df_demo tiene NaN — OK para imputers

    # ── 2a: MAE por metodo ────────────────────────────────────────────────────────
    print(f"\n2a. MAE en los {mask_nan_bool.sum()} registros con NaN artificial (MAR):")
    pred_guardadas = {}  # guardar para 2b

    for nombre, imp in [
        ('SimpleImputer', SimpleImputer(strategy='median')),
        ('KNNImputer k=5', KNNImputer(n_neighbors=5)),
        ('MICE',          IterativeImputer(max_iter=10, random_state=42)),
    ]:
        if 'Simple' in nombre:
            # SimpleImputer trabaja directo sobre el DataFrame
            X_imp = imp.fit_transform(df_demo)
        else:
            # KNN y MICE trabajan sobre los datos normalizados
            X_imp_sc = imp.fit_transform(X_sc)
            X_imp = scaler.inverse_transform(X_imp_sc)

        # Extraer prima_neta imputada como array numpy
        idx_prima = features_num.index('prima_neta')
        pred_arr  = X_imp[:, idx_prima]

        # Calcular MAE solo donde habia NaN — usando mascara booleana numpy
        mae = mean_absolute_error(
            prima_verdadera_arr[mask_nan_bool],
            pred_arr[mask_nan_bool]
        )
        print(f"  {nombre:<20}: MAE=${mae:,.2f}")
        pred_guardadas[nombre] = pred_arr

    # ── 2b: Scatter real vs imputado ──────────────────────────────────────────────
    print("\n2b. Generando scatter real vs imputado...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True, sharex=True)
    colores = {'SimpleImputer':'#C0392B', 'KNNImputer k=5':'#1A5276', 'MICE':'#1E8449'}

    for ax, (nombre, pred_arr) in zip(axes, pred_guardadas.items()):
        real_f = prima_verdadera_arr[mask_nan_bool]
        pred_f = pred_arr[mask_nan_bool]
        mae_v  = mean_absolute_error(real_f, pred_f)

        ax.scatter(real_f, pred_f, alpha=0.4, s=15, color=colores[nombre])

        lim = max(real_f.max(), pred_f.max())
        ax.plot([0, lim], [0, lim], color='red', linestyle='--',
                linewidth=1.5, label='Ajuste perfecto (y=x)')

        ax.set_title(f'{nombre}\nMAE=${mae_v:,.0f}', fontweight='bold')
        ax.set_xlabel('Prima real (MXN)')
        if ax == axes[0]:
            ax.set_ylabel('Prima imputada (MXN)')
        ax.legend(fontsize=8)

    fig.suptitle(
        'Real vs Imputado — Registros con NaN artificial (MAR)\n'
        'Puntos sobre la diagonal roja = imputacion perfecta',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.show()
    print("  SimpleImputer: puntos agrupados en linea horizontal (siempre el mismo valor)")
    print("  KNN y MICE:    puntos mas cerca de la diagonal (respetan la variacion real)")

    # ── 2c: Comparar distintos k ──────────────────────────────────────────────────
    print("\n2c. KNNImputer con distintos k:")
    for k in [3, 5, 10, 20]:
        X_knn    = KNNImputer(n_neighbors=k).fit_transform(X_sc)
        pred_arr = scaler.inverse_transform(X_knn)[:, features_num.index('prima_neta')]
        mae      = mean_absolute_error(
            prima_verdadera_arr[mask_nan_bool],
            pred_arr[mask_nan_bool]
        )
        print(f"  k={k:>2}: MAE=${mae:,.2f}")

    print()
    print("2d. Usar MICE cuando hay varias columnas con NaN correlacionadas entre si.")
    print("    En la cartera: prima_neta y deducible tienen correlacion — MICE es mejor.")


def solucion_final():
    print("="*60)
    print("  SOLUCION INTEGRADOR — Pipeline Completo")
    print("="*60)
    features_modelo = ['edad','suma_asegurada','deducible','n_siniestros']
    df_modelo = df_raw[features_modelo+['prima_total']].dropna(subset=['prima_total'])
    X = df_modelo[features_modelo]
    y = df_modelo['prima_total']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\nFase 4 — Impacto en el modelo (LinearRegression):")
    for nombre, imp in [
        ('SimpleImputer (median)', SimpleImputer(strategy='median')),
        ('KNNImputer (k=5)',       KNNImputer(n_neighbors=5)),
        ('IterativeImputer',       IterativeImputer(max_iter=10, random_state=42)),
    ]:
        pipe = Pipeline([('imp', imp), ('sc', StandardScaler()), ('mod', LinearRegression())])
        pipe.fit(X_tr, y_tr)
        yp   = pipe.predict(X_te)
        print(f"  {nombre:<28}: R2={r2_score(y_te,yp):.4f}  MAE=${mean_absolute_error(y_te,yp):,.0f}")
    print()
    print("Conclusion: en este caso la diferencia entre metodos es pequena.")
    print("La imputacion importa mas cuando el % de NaN es alto (>10%).")
    print("Para prima_neta (~2% NaN) cualquier metodo razonable funciona bien.")


if __name__=='__main__':
    arg = sys.argv[1].lower() if len(sys.argv)>1 else 'all'
    if arg in ('all','e1'):    solucion_e1()
    if arg in ('all','e2'):    solucion_e2()
    if arg in ('all','final'): solucion_final()
