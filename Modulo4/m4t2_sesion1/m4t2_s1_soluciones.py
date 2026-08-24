"""
m4t2_s1_soluciones.py
=====================
Soluciones a los ejercicios de la Sesión 1 — Módulo 4, Tema 2 (GLM con Python).
Diplomado: Machine Learning en Seguros · FC UNAM · 22 de agosto de 2026

Prerequisito: datos/datos.pkl ya generado.

Uso:
    python m4t2_s1_soluciones.py
    # o:  from m4t2_s1_soluciones import solucion_ejercicio1
"""

import os
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

pd.set_option('display.float_format', '{:,.4f}'.format)
pd.set_option('display.max_columns', 25); pd.set_option('display.width', 120)
RUTA = 'datos/datos.pkl'


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS (los mismos del notebook)
# ─────────────────────────────────────────────────────────────────────────────
def cargar():
    d = pd.read_pickle(RUTA)
    d['frecuencia'] = d['num_siniestros'] / d['exposicion']
    d['freq_ref'] = d['num_siniestros'].sum() / d['exposicion'].sum()
    return d


def tasa_empirica(df, variable, min_expo=0):
    """Tasa de siniestralidad ponderada por exposición, con IC 95%."""
    g = df.groupby(variable, observed=True)
    out = g.apply(lambda x: pd.Series({
        'expo':  x['exposicion'].sum(),
        'sinis': x['num_siniestros'].sum(),
        'freq':  x['num_siniestros'].sum() / x['exposicion'].sum(),
        's2':    np.average((x['num_siniestros']/x['exposicion'] -
                             x['num_siniestros'].sum()/x['exposicion'].sum())**2,
                            weights=x['exposicion']),
    }), include_groups=False).reset_index()
    out['ic'] = 1.96 * np.sqrt(out['s2']) / np.sqrt(out['expo'])
    return out[out['expo'] >= min_expo]


def phi_pearson(m):
    return np.sum(m.resid_pearson**2) / m.df_resid


def cameron_trivedi(m, df):
    """Prueba de sobredispersión: OLS de (y-μ)²-y sobre μ (sin intercepto)."""
    mu = m.fittedvalues
    u = (df['num_siniestros'] - mu)**2 - df['num_siniestros']
    aux = sm.OLS(u, mu).fit()
    return aux.params.iloc[0], aux.tvalues.iloc[0], aux.pvalues.iloc[0]


def rating_factors(m):
    return pd.DataFrame({
        'coef': m.params, 'p_value': m.pvalues,
        'RF': np.exp(m.params),
        'RF_lower': np.exp(m.params - 1.96*m.bse),
        'RF_upper': np.exp(m.params + 1.96*m.bse),
    })


BANDINGS = {
    'edad_conductor':      [17, 30, 35, 45, 50, 55, 60, 95],
    'antiguedad_vehiculo': [-1, 1, 2, 3, 4, 5, 10, 15, 50],
    'potencia':            [9, 40, 50, 60, 70, 250],
    'nivel_bonus':         [-1, 0, 1, 2, 5, 10, 25],
}
def con_bandings(d):
    for c, ct in BANDINGS.items():
        d[c + '_cat'] = pd.cut(d[c], bins=ct).astype(str)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# EJERCICIO 1 — Tasa empírica de nivel_bonus con IC
# ─────────────────────────────────────────────────────────────────────────────
def solucion_ejercicio1():
    print("=" * 65); print("  EJERCICIO 1 — Tasa empírica de nivel_bonus"); print("=" * 65)
    d = cargar()

    te = tasa_empirica(d, 'nivel_bonus')
    print("\n1a. Tasa empírica con IC 95% por nivel_bonus:")
    print(te[['nivel_bonus', 'expo', 'freq', 'ic']].round(4).to_string(index=False))

    print("\n1b. La tendencia SÍ tiene sentido actuarial: el bonus-malus es un score de")
    print("    siniestralidad histórica, así que a mayor nivel esperamos mayor frecuencia")
    print("    (el score ya venía capturando riesgo).")

    ancho = te.sort_values('ic', ascending=False)
    print("\n1c. Los IC más anchos están en:")
    print(ancho[['nivel_bonus', 'expo', 'ic']].head(3).round(4).to_string(index=False))
    print("    → coinciden con los niveles de MENOR exposición: menos años-póliza,")
    print("      raíz más grande en el denominador, banda más ancha.")
    print()
    return te


# ─────────────────────────────────────────────────────────────────────────────
# EJERCICIO 2 — Banding de potencia con exposición por banda
# ─────────────────────────────────────────────────────────────────────────────
def solucion_ejercicio2():
    print("=" * 65); print("  EJERCICIO 2 — Banding de potencia"); print("=" * 65)
    d = cargar()
    expo_total = d['exposicion'].sum()

    # 2a: propuesta de banding (criterio: cortes con exposición suficiente)
    cortes = [9, 40, 50, 60, 70, 250]
    d['potencia_cat'] = pd.cut(d['potencia'], bins=cortes)
    r = d.groupby('potencia_cat', observed=True).agg(
        expo=('exposicion', 'sum'), sinis=('num_siniestros', 'sum')).reset_index()
    r['freq'] = r['sinis'] / r['expo']
    r['pct_expo'] = r['expo'] / expo_total * 100

    print(f"\n2a-b. Banding propuesto {cortes} y exposición por banda:")
    print(r[['potencia_cat', 'expo', 'pct_expo', 'freq']].round(4).to_string(index=False))
    minima = r.loc[r['pct_expo'].idxmin()]
    print(f"\n     Banda con menos exposición: {minima['potencia_cat']} "
          f"({minima['pct_expo']:.1f}% del total).")
    if (r['pct_expo'] < 1).any():
        print("     ⚠ Hay bandas con < 1% de exposición: conviene fusionarlas con la vecina.")
    else:
        print("     ✓ Todas las bandas superan el 1% de exposición: banding estable.")

    print("\n2c. Criterio: cortar donde la curva de frecuencia cambia de pendiente,")
    print("    pero exigiendo exposición suficiente por banda. Un banding más fino")
    print("    daría más resolución pero cada banda sería más ruidosa (IC más ancho).")
    print()
    return r


# ─────────────────────────────────────────────────────────────────────────────
# EJERCICIO 3 — Offset ↔ weights con C(uso)
# ─────────────────────────────────────────────────────────────────────────────
def solucion_ejercicio3():
    print("=" * 65); print("  EJERCICIO 3 — Offset ↔ weights (uso)"); print("=" * 65)
    d = cargar()

    m_off = smf.glm('num_siniestros ~ C(uso)', d, family=sm.families.Poisson(),
                    offset=np.log(d['exposicion'])).fit()
    m_wei = smf.glm('frecuencia ~ C(uso)', d, family=sm.families.Poisson(),
                    freq_weights=d['exposicion']).fit()

    comp = pd.DataFrame({'offset': m_off.params, 'weights': m_wei.params})
    comp['dif_abs'] = (comp['offset'] - comp['weights']).abs()
    print("\n3a-b. Coeficientes lado a lado:")
    print(comp.to_string())
    print(f"\n     Diferencia máxima: {comp['dif_abs'].max():.2e}  (≈ 0 → equivalentes)")

    print("\n3c. Se prefiere el OFFSET porque modela el conteo real (devianza y AIC")
    print("    interpretables como conteo), evita crear frecuencia = N/E —inestable")
    print("    con exposiciones diminutas— y es el estándar N ~ Poisson(E·λ) para CNSF.")
    print()
    return m_off, m_wei


# ─────────────────────────────────────────────────────────────────────────────
# EJERCICIO 4 — ¿Qué familia para qué cartera?
# ─────────────────────────────────────────────────────────────────────────────
def solucion_ejercicio4():
    print("=" * 65); print("  EJERCICIO 4 — Qué familia para qué cartera"); print("=" * 65)
    d = con_bandings(cargar())

    print("\n4a. Deducible muy alto, 85% nunca reclama (exceso de ceros ESTRUCTURAL):")
    print("    → ZIP o Hurdle. La subpoblación que 'nunca reclama' se modela aparte del")
    print("      proceso Poisson. Si además el reporte CNSF exige separar 'sí/no reclama'")
    print("      de 'cuántos', se prefiere HURDLE (dos etapas explícitas).")

    print("\n4b. Gastos médicos con heterogeneidad marcada (φ ≈ 3, sobredispersión fuerte):")
    print("    → Binomial Negativa. Modela heterogeneidad no observada vía la mezcla")
    print("      Poisson-Gamma, V(μ)=μ+μ²/α, sin necesidad de variables nuevas.")

    print("\n4c. Daños con sobredispersión leve (φ ≈ 1.3) y CNSF pide explicar P(siniestro):")
    print("    → Hurdle: la etapa logit modela directamente la probabilidad de siniestro,")
    print("      fácil de documentar; y la etapa truncada modela la intensidad.")

    # Ilustración empírica: ajustamos Poisson vs BN y comparamos por AIC
    print("\n    Ilustración sobre la cartera real (Poisson vs Binomial Negativa):")
    m_p = smf.glm('num_siniestros ~ C(edad_conductor_cat) + C(nivel_bonus_cat)', d,
                  family=sm.families.Poisson(), offset=np.log(d['exposicion'])).fit()
    m_nb = smf.glm('num_siniestros ~ C(edad_conductor_cat) + C(nivel_bonus_cat)', d,
                   family=sm.families.NegativeBinomial(alpha=1.0),
                   offset=np.log(d['exposicion'])).fit()
    phi = phi_pearson(m_p)
    print(f"      φ = {phi:.3f}  ·  AIC Poisson = {m_p.aic:,.0f}  ·  AIC BN = {m_nb.aic:,.0f}")
    print(f"      Gana: {'Binomial Negativa' if m_nb.aic < m_p.aic else 'Poisson'} "
          f"(pero con φ≈{phi:.2f}, la sobredispersión es leve → QuasiPoisson también sirve).")
    print()
    return m_p, m_nb


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRADOR — Modelo de frecuencia de principio a fin
# ─────────────────────────────────────────────────────────────────────────────
def solucion_integrador():
    print("=" * 65); print("  INTEGRADOR — Frecuencia de principio a fin"); print("=" * 65)
    d = con_bandings(cargar())

    # 1) EDA empírica (una muestra)
    print("\n[1] EDA empírica (cobertura) con IC 95%:")
    print(tasa_empirica(d, 'cobertura')[['cobertura', 'expo', 'freq', 'ic']].round(4).to_string(index=False))

    # 2) bandings ya aplicados (revisión de exposición mínima)
    print("\n[2] Bandings aplicados. Exposición mínima por banda:")
    for c in BANDINGS:
        mn = d.groupby(c + '_cat', observed=True)['exposicion'].sum().min()
        print(f"    {c:<22}: {mn:,.0f} años-póliza")

    # 3) modelo completo
    formula = ('num_siniestros ~ C(cobertura) + C(sexo) + C(uso) + C(combustible) + '
               'C(edad_conductor_cat) + C(antiguedad_vehiculo_cat) + '
               'C(potencia_cat) + C(nivel_bonus_cat)')
    m = smf.glm(formula, d, family=sm.families.Poisson(), offset=np.log(d['exposicion'])).fit()
    print(f"\n[3] Modelo completo: {len(m.params)} parámetros · AIC = {m.aic:,.0f} · "
          f"Deviance = {m.deviance:,.0f}")

    # 4) diagnóstico
    phi = phi_pearson(m); alpha, z, pval = cameron_trivedi(m, d)
    print(f"\n[4] Diagnóstico: φ = {phi:.4f}  ·  Cameron-Trivedi α = {alpha:.4f} "
          f"(z = {z:.1f}, p = {pval:.2g})")
    if phi < 1.5:
        print("    → Sobredispersión leve: Poisson con QuasiPoisson para la inferencia basta.")
    else:
        m_nb = smf.glm(formula, d, family=sm.families.NegativeBinomial(alpha=1.0),
                       offset=np.log(d['exposicion'])).fit()
        print(f"    → φ elevado: comparamos con BN. AIC Poisson={m.aic:,.0f}, AIC BN={m_nb.aic:,.0f}")

    # 5) rating factors con IC → CSV
    rf = rating_factors(m); rf.to_csv('rating_factors_frecuencia.csv')
    print(f"\n[5] Rating factors con IC 95% → rating_factors_frecuencia.csv ({len(rf)} filas)")

    # 6) puente empírico
    m_ow = smf.glm('num_siniestros ~ C(cobertura)', d, family=sm.families.Poisson(),
                   offset=np.log(d['exposicion'])).fit()
    base = np.exp(m_ow.params['Intercept'])
    niveles = sorted(d['cobertura'].unique())
    ref = [n for n in niveles if f'C(cobertura)[T.{n}]' not in m_ow.params.index][0]
    dif_max = 0
    for n in niveles:
        rf_n = 1.0 if n == ref else np.exp(m_ow.params[f'C(cobertura)[T.{n}]'])
        emp = d.loc[d.cobertura == n, 'num_siniestros'].sum() / d.loc[d.cobertura == n, 'exposicion'].sum()
        dif_max = max(dif_max, abs(base*rf_n - emp))
    print(f"\n[6] Puente empírico (cobertura): |base×RF − tasa empírica| máx = {dif_max:.2e}")
    print("    El GLM one-way reproduce la tasa empírica al centavo.")

    # 7) documentación
    rf_sig = rf.drop('Intercept').copy(); rf_sig['desvio'] = (rf_sig['RF'] - 1).abs()
    top2 = rf_sig.sort_values('desvio', ascending=False).head(2)
    print("\n[7] Los dos rating factors más fuertes (para nota técnica):")
    for term, row in top2.iterrows():
        pct = (row['RF'] - 1) * 100
        signo = 'recargo' if pct >= 0 else 'descuento'
        print(f"    - {term}")
        print(f"        RF = {row['RF']:.3f} ({abs(pct):.0f}% de {signo}) · "
              f"IC95% [{row['RF_lower']:.3f}, {row['RF_upper']:.3f}] · p = {row['p_value']:.2g}")
    print()
    return m, rf


if __name__ == '__main__':
    print("\nSESIÓN 1 — SOLUCIONES COMPLETAS")
    print("Diplomado ML en Seguros · FC UNAM · 22 de agosto 2026"); print("=" * 65)
    if not os.path.exists(RUTA):
        print(f"ERROR: no existe {RUTA}."); raise SystemExit(1)
    solucion_ejercicio1()
    solucion_ejercicio2()
    solucion_ejercicio3()
    solucion_ejercicio4()
    solucion_integrador()
    print("Todas las soluciones se ejecutaron correctamente.")
