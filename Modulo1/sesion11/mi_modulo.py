"""
Modulo de funciones actuariales para el Diplomado ML en Seguros.
Facultad de Ciencias, UNAM — Modulo 1.
"""

# ── Funciones de primas ──────────────────────────────────────────────────
def calcular_prima(suma_asegurada, tasa_riesgo, recargo=0.15, iva=1.16):
    """Calcula la prima comercial de una poliza de seguros."""
    prima_pura = suma_asegurada * tasa_riesgo
    return prima_pura * (1 + recargo) * iva

def prima_mensual(suma_asegurada, tasa_riesgo, **kwargs):
    """Devuelve la prima mensual dividiendo entre 12."""
    return calcular_prima(suma_asegurada, tasa_riesgo, **kwargs) / 12

# ── Funciones de clasificacion ───────────────────────────────────────────
def clasificar_riesgo(siniestros):
    """Clasifica nivel de riesgo por numero de siniestros.
    Retorna: 'ALTO', 'MEDIO' o 'BAJO'
    """
    if siniestros >= 3:   return 'ALTO'
    elif siniestros >= 1: return 'MEDIO'
    return 'BAJO'

def grupo_edad(edad):
    """Devuelve el grupo tarifario por edad."""
    if edad <= 30:   return '18-30'
    elif edad <= 45: return '31-45'
    elif edad <= 60: return '46-60'
    return '61+'

# ── Metricas de cartera ───────────────────────────────────────────────────
def metricas_cartera(*primas):
    """Calcula metricas basicas de una cartera.
    Retorna dict con n, total, promedio, maximo, minimo.
    """
    if not primas: return None
    return dict(
        n=len(primas), total=sum(primas),
        promedio=sum(primas)/len(primas),
        maximo=max(primas), minimo=min(primas)
    )