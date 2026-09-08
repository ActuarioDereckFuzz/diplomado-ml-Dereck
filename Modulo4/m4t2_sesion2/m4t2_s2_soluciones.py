"""
m4t2_s2_soluciones.py
=====================
Soluciones a los ejercicios de la Sesión 2 — Módulo 4, Tema 2 (GLM con Python).
Diplomado ML en Seguros · FC UNAM · 26 de agosto de 2026

Prerequisito: datos/datos.pkl.
Uso:  python m4t2_s2_soluciones.py
"""
import os, json
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf
from sklearn.linear_model import TweedieRegressor

pd.set_option('display.float_format','{:,.4f}'.format); pd.set_option('display.width',120)
RUTA = 'datos/datos.pkl'
BANDINGS = {'edad_conductor':[17,30,35,45,50,55,60,95],'antiguedad_vehiculo':[-1,1,2,3,4,5,10,15,50],
            'potencia':[9,40,50,60,70,250],'nivel_bonus':[-1,0,1,2,5,10,25]}
GAMMA = sm.families.Gamma(sm.families.links.Log())


def cargar():
    d = pd.read_pickle(RUTA)
    for c,ct in BANDINGS.items(): d[c+'_cat'] = pd.cut(d[c],bins=ct).astype(str)
    d['pure_premium'] = d['monto_total']/d['exposicion']
    return d, d[d['num_siniestros']>0].copy()


def severidad_empirica(df, variable):
    g = df.groupby(variable, observed=True)
    o = g.apply(lambda x: pd.Series({
        'nclaims':x['num_siniestros'].sum(),
        'sev':np.average(x['severidad'],weights=x['num_siniestros']),
        'std':np.sqrt(np.average((x['severidad']-np.average(x['severidad'],weights=x['num_siniestros']))**2,weights=x['num_siniestros']))
    }), include_groups=False).reset_index()
    o['CV'] = o['std']/o['sev']; return o


def pseudo_r2(modelo, formula_nula, data, **kw):
    m0 = smf.glm(formula_nula, data, family=modelo.family, **kw).fit()
    return {'McFadden':1-modelo.llf/m0.llf, 'devianza':1-modelo.deviance/m0.deviance}


def forward_stepwise(data, respuesta, candidatas, offset=None, family=None, weights=None, verbose=False):
    family = family or sm.families.Poisson()
    kw = {} if weights is None else {'freq_weights':weights}
    sel, aic = [], smf.glm(f'{respuesta} ~ 1', data, family=family, offset=offset, **kw).fit().aic
    while True:
        best = None
        for v in candidatas:
            if v in sel: continue
            a = smf.glm(f'{respuesta} ~ '+' + '.join(sel+[v]), data, family=family, offset=offset, **kw).fit().aic
            if best is None or a < best[1]: best = (v, a)
        if best is None or best[1] >= aic: break
        sel.append(best[0]); aic = best[1]
        if verbose: print(f'  + {best[0]:<28} AIC = {aic:,.1f}')
    return sel


# ── EJERCICIO 1 — CV constante por edad ──────────────────────────────────────
def solucion_ejercicio1():
    print("="*65); print("  EJERCICIO 1 — CV constante por edad"); print("="*65)
    d, sev = cargar()
    cv = severidad_empirica(sev, 'edad_conductor_cat')
    print("\n1a. Severidad y CV por edad:")
    print(cv.round(3).to_string(index=False))
    print(f"\n1b. CV: media={cv['CV'].mean():.3f}, desv={cv['CV'].std():.3f}.")
    print("    Se mantiene razonablemente estable entre grupos → justifica Gamma")
    print("    (variabilidad relativa parecida en todos los niveles de severidad).")
    g = smf.glm('severidad ~ C(edad_conductor_cat)', sev, family=GAMMA, freq_weights=sev['num_siniestros']).fit()
    print(f"\n1c. √escala del modelo (CV implícito): {np.sqrt(g.scale):.3f}")
    print()
    return cv, g


# ── EJERCICIO 2 — Sensibilidad al umbral de capping ──────────────────────────
def solucion_ejercicio2():
    print("="*65); print("  EJERCICIO 2 — Sensibilidad del capping"); print("="*65)
    d, sev = cargar()
    total = sev['monto_total'].sum()
    base = np.average(sev['severidad'], weights=sev['num_siniestros'])
    print(f"\nSeveridad media SIN capping: ${base:,.0f}\n")
    print(f"{'umbral':>8} {'sev media':>12} {'% cedido reaseguro':>20}")
    for q in [95, 99, 99.5]:
        cap = np.percentile(sev['monto_total'], q)
        montos_cap = np.minimum(sev['monto_total'], cap)
        sev_cap = np.average(montos_cap/sev['num_siniestros'], weights=sev['num_siniestros'])
        cedido = (sev['monto_total']-montos_cap).sum()/total
        print(f"P{q:<7} ${sev_cap:>10,.0f} {cedido:>19.1%}")
    print("\n2c. Un umbral más bajo (P95) estabiliza más pero cede más monto a reaseguro")
    print("    y pierde información de siniestros medianos legítimos. El P99 suele ser")
    print("    el balance estándar: quita solo la cola extrema. Documentar en nota técnica.")
    print()


# ── EJERCICIO 3 — pseudo R² de severidad ─────────────────────────────────────
def solucion_ejercicio3():
    print("="*65); print("  EJERCICIO 3 — pseudo R² de severidad"); print("="*65)
    d, sev = cargar()
    w = sev['num_siniestros']
    m_full = smf.glm('severidad ~ C(cobertura)+C(edad_conductor_cat)+C(potencia_cat)', sev, family=GAMMA, freq_weights=w).fit()
    m_red  = smf.glm('severidad ~ C(cobertura)', sev, family=GAMMA, freq_weights=w).fit()
    r_full = pseudo_r2(m_full, 'severidad ~ 1', sev, freq_weights=w)
    r_red  = pseudo_r2(m_red,  'severidad ~ 1', sev, freq_weights=w)
    print(f"\n3a. Modelo completo: pseudo R² McFadden={r_full['McFadden']:.4f}, devianza={r_full['devianza']:.4f}")
    print(f"    Modelo reducido: pseudo R² McFadden={r_red['McFadden']:.4f}, devianza={r_red['devianza']:.4f}")
    print(f"\n3b. AIC completo={m_full.aic:,.0f}  ·  AIC reducido={m_red.aic:,.0f}  → "
          f"gana {'completo' if m_full.aic<m_red.aic else 'reducido'}")
    print("\n3c. El pseudo R² de severidad suele ser AÚN más bajo que el de frecuencia:")
    print("    el tamaño de un siniestro depende de factores casi irreducibles (azar del")
    print("    evento), más difíciles de explicar con variables de la póliza.")
    print()


# ── EJERCICIO 4 — stepwise para severidad ────────────────────────────────────
def solucion_ejercicio4():
    print("="*65); print("  EJERCICIO 4 — Stepwise para severidad"); print("="*65)
    d, sev = cargar()
    cand = ['C(cobertura)','C(sexo)','C(uso)','C(combustible)','C(edad_conductor_cat)',
            'C(antiguedad_vehiculo_cat)','C(potencia_cat)','C(nivel_bonus_cat)']
    print("\n4a. Forward stepwise (severidad, Gamma, weights=N):")
    sel_sev = forward_stepwise(sev, 'severidad', cand, family=GAMMA, weights=sev['num_siniestros'], verbose=True)
    print("    Seleccionadas (severidad):", sel_sev)

    sel_freq = forward_stepwise(d, 'num_siniestros', cand, offset=np.log(d['exposicion']))
    print("\n4b. Seleccionadas (frecuencia):", sel_freq)
    print(f"    Coinciden en: {sorted(set(sel_sev)&set(sel_freq))}")
    print(f"    Solo en severidad: {sorted(set(sel_sev)-set(sel_freq))}")
    print(f"    Solo en frecuencia: {sorted(set(sel_freq)-set(sel_sev))}")
    print("\n4c. Las variables de 'cuántos' siniestros (edad, bonus) no tienen por qué")
    print("    predecir 'de qué tamaño'. La cobertura y la potencia suelen pesar más en")
    print("    severidad; el bonus-malus más en frecuencia. Por eso Freq×Sev usa modelos")
    print("    separados: cada componente tiene su propia estructura de riesgo.")
    print()


# ── INTEGRADOR — elige y guarda los mejores modelos ──────────────────────────
def solucion_integrador():
    print("="*65); print("  INTEGRADOR — Elige y guarda tu mejor tarifa"); print("="*65)
    d, sev = cargar()
    os.makedirs('modelos', exist_ok=True)
    cand = ['C(cobertura)','C(sexo)','C(uso)','C(combustible)','C(edad_conductor_cat)',
            'C(antiguedad_vehiculo_cat)','C(potencia_cat)','C(nivel_bonus_cat)']

    # 1) stepwise freq y sev
    sel_f = forward_stepwise(d, 'num_siniestros', cand, offset=np.log(d['exposicion']))
    sel_s = forward_stepwise(sev, 'severidad', cand, family=GAMMA, weights=sev['num_siniestros'])
    f_formula = 'num_siniestros ~ '+' + '.join(sel_f)
    s_formula = 'severidad ~ '+' + '.join(sel_s)
    m_f = smf.glm(f_formula, d, family=sm.families.Poisson(), offset=np.log(d['exposicion'])).fit()
    m_s = smf.glm(s_formula, sev, family=GAMMA, freq_weights=sev['num_siniestros']).fit()
    print(f"\n[1] Frecuencia: {len(m_f.params)} params · AIC {m_f.aic:,.0f}")
    print(f"    Severidad : {len(m_s.params)} params · AIC {m_s.aic:,.0f}")

    # 2) Tweedie con p óptimo
    feats = ['cobertura','sexo','combustible','edad_conductor_cat','potencia_cat','nivel_bonus_cat']
    X = pd.get_dummies(d[feats], drop_first=True).astype(float).values
    y, w = d['pure_premium'].values, d['exposicion'].values
    p_opt = max([1.2,1.3,1.4,1.5,1.6,1.7],
                key=lambda pp: TweedieRegressor(power=pp,alpha=0,link='log',max_iter=400).fit(X,y,sample_weight=w).score(X,y,sample_weight=w))
    tw_formula = 'pure_premium ~ '+' + '.join('C('+f+')' for f in feats)
    m_t = smf.glm(tw_formula, d, family=sm.families.Tweedie(var_power=p_opt, link=sm.families.links.Log()),
                  var_weights=d['exposicion']).fit()
    pp_obs = (d['pure_premium']*d['exposicion']).sum(); pp_tw = (m_t.predict(d)*d['exposicion']).sum()
    print(f"[2] Tweedie p={p_opt}: prima pura total ratio Tw/obs = {pp_tw/pp_obs:.4f}")

    # 3) guardar
    meta = {'frecuencia':{'respuesta':'num_siniestros','formula':f_formula,'family':'Poisson','offset':'log(exposicion)'},
            'severidad':{'respuesta':'severidad','formula':s_formula,'family':'Gamma(log)','weights':'num_siniestros','filtro':'num_siniestros>0'},
            'agregado':{'respuesta':'pure_premium','formula':tw_formula,'family':'Tweedie','p':float(p_opt),'weights':'exposicion'},
            'bandings':BANDINGS}
    json.dump(meta, open('modelos/metadatos.json','w'), indent=2, ensure_ascii=False)
    print("[3-4] Guardado en modelos/metadatos.json (la Sesión 3 lo reconstruye).")

    print("\n[5] Freq × Sev vs Tweedie para nota técnica CNSF:")
    print("    Freq × Sev da rating factors SEPARADOS por componente (más trazable,")
    print("    preferido por la industria mexicana). Tweedie es un benchmark rápido y")
    print("    útil cuando solo tienes prima pura. Para la nota técnica: Freq × Sev como")
    print("    modelo principal, Tweedie como validación complementaria.")
    print()


if __name__ == '__main__':
    print("\nSESIÓN 2 — SOLUCIONES COMPLETAS")
    print("Diplomado ML en Seguros · FC UNAM · 26 de agosto 2026"); print("="*65)
    if not os.path.exists(RUTA):
        print(f"ERROR: no existe {RUTA}."); raise SystemExit(1)
    solucion_ejercicio1(); solucion_ejercicio2(); solucion_ejercicio3()
    solucion_ejercicio4(); solucion_integrador()
    print("Todas las soluciones se ejecutaron correctamente.")
