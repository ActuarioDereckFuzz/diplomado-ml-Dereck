"""
cargar_modelos.py
=================
Reconstruye los tres mejores modelos elegidos en la Sesión 2 (frecuencia,
severidad y agregado/Tweedie) a partir de modelos/metadatos.json y datos/datos.pkl.

Refit exacto y ligero (~1-2 s). Lo usa la Sesión 3 para visualizar efectos, validar,
hacer análisis de sensibilidad y simular la tarifa.

Uso:
    from cargar_modelos import cargar_modelos, preparar_datos
    datos, sev = preparar_datos()
    modelos = cargar_modelos()            # dict: 'frecuencia','severidad','agregado'
    modelos['frecuencia'].summary()
"""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings; warnings.filterwarnings('ignore')


RUTA_DATOS = '../m4t2_sesion1/datos/datos.pkl'
RUTA_META  = '../m4t2_sesion2/modelos/metadatos.json'


def preparar_datos(ruta=RUTA_DATOS, ruta_meta=RUTA_META):
    """Carga la cartera y aplica los bandings guardados. Devuelve (datos, sev)."""
    meta = json.load(open(ruta_meta, encoding='utf-8'))
    datos = pd.read_pickle(ruta)
    for col, cortes in meta['bandings'].items():
        datos[col + '_cat'] = pd.cut(datos[col], bins=cortes).astype(str)
    datos['pure_premium'] = datos['monto_total'] / datos['exposicion']
    sev = datos[datos['num_siniestros'] > 0].copy()   # severidad: solo N>0
    return datos, sev


def cargar_modelos(ruta=RUTA_DATOS, ruta_meta=RUTA_META, verbose=False):
    """Reconstruye los 3 modelos por refit desde sus fórmulas guardadas."""
    meta = json.load(open(ruta_meta, encoding='utf-8'))
    datos, sev = preparar_datos(ruta, ruta_meta)

    f = meta['frecuencia']
    m_freq = smf.glm(f['formula'], datos, family=sm.families.Poisson(),
                     offset=np.log(datos['exposicion'])).fit()

    s = meta['severidad']
    m_sev = smf.glm(s['formula'], sev, family=sm.families.Gamma(sm.families.links.Log()),
                    freq_weights=sev['num_siniestros']).fit()

    a = meta['agregado']
    m_agg = smf.glm(a['formula'], datos,
                    family=sm.families.Tweedie(var_power=a['p'], link=sm.families.links.Log()),
                    var_weights=datos['exposicion']).fit()

    if verbose:
        print(f"frecuencia: {len(m_freq.params)} params · AIC {m_freq.aic:,.0f}")
        print(f"severidad : {len(m_sev.params)} params · AIC {m_sev.aic:,.0f}")
        print(f"agregado  : Tweedie p={a['p']} · {len(m_agg.params)} params")
    return {'frecuencia': m_freq, 'severidad': m_sev, 'agregado': m_agg, 'meta': meta}


if __name__ == '__main__':
    import os
    if not (os.path.exists(RUTA_META) and os.path.exists(RUTA_DATOS)):
        print('Falta modelos/metadatos.json o datos/datos.pkl. Corre primero la Sesión 2.')
        raise SystemExit(1)
    m = cargar_modelos(verbose=True)
    print('\nModelos reconstruidos correctamente:', list(m)[:3])
