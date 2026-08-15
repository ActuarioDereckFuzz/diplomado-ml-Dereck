"""
Generador de cartera sintética ampliada v2 — Subtema 4, Módulo 3 Tema 1.

Función principal: simulate_auto_insurance_portfolio_v2()
Schema: 21 columnas × n_policies filas.
Semilla base del proyecto: 20260618.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ── Constantes exportables — reutilizables en notebooks y scripts ─────────────

ALL_COLUMNS: list[str] = [
    "id_poliza_sin",        # col 1  — identificador (nunca feature)
    "anio_poliza",          # col 2  — feature temporal
    "mes_emision",          # col 3  — feature temporal
    "edad",                 # col 4  — feature
    "sexo",                 # col 5  — feature
    "estado_civil",         # col 6  — feature
    "num_renovaciones",     # col 7  — feature (estructura de cluster)
    "region",               # col 8  — feature
    "plan",                 # col 9  — feature (estructura de cluster)
    "tipo_vehiculo",        # col 10 — feature
    "antiguedad_vehiculo",  # col 11 — feature (estructura de cluster)
    "suma_asegurada",       # col 12 — feature (estructura de cluster)
    "deducible_pct",        # col 13 — feature
    "forma_pago",           # col 14 — feature
    "canal_venta",          # col 15 — feature
    "nivel_riesgo_asignado",# col 16 — feature derivada de suscripción
    "prima_neta",           # col 17 — target regresión
    "n_siniestros",         # col 18 — target conteo
    "monto_siniestro",      # col 19 — target regresión condicional
    "siniestro_alto",       # col 20 — target clasificación binaria
    "status_poliza",        # col 21 — target clasificación multiclase
    "kilometraje",           # col 22 — target clasificación multiclase
    "uso_vehiculo",          # col 23 — target clasificación multiclase
    "color_vehiculo"        # col 24 — target clasificación multiclase
]

# Features: columnas 2–16 (índice 1–15); no incluye id_poliza_sin ni targets
FEATURE_COLUMNS: list[str] = ALL_COLUMNS[1:16] + ["kilometraje",'color_vehiculo','uso_vehiculo']

# Targets: columnas 17–21 (índice 16–20); nunca se usan como features (§2.2)
TARGET_COLUMNS: list[str] = ALL_COLUMNS[16:21]

# Features permitidas para clustering (subconjunto de FEATURE_COLUMNS, §2.2)
CLUSTERING_FEATURES: list[str] = [
    "edad",
    "suma_asegurada",
    "num_renovaciones",
    "antiguedad_vehiculo",
]

# Columnas PII prohibidas — ninguna debe aparecer en ALL_COLUMNS (§AGENTS.md)
PII_COLUMNS_FORBIDDEN: list[str] = [
    "rfc",
    "nombre",
    "apellido_paterno",
    "apellido_materno",
    "nombre_agente",
]


def simulate_auto_insurance_portfolio_v2(
    n_policies: int = 30_000,
    seed: int = 20260618,
    include_quality_noise: bool = False,
) -> pd.DataFrame:
    """
    Genera una cartera sintética ampliada de seguros de autos para el Subtema 4.

    Diseñada para ilustrar 11 técnicas de ML (regresión, clasificación, clustering)
    con relaciones causales controladas, sin PII, sin leakage estructural y con
    cero nulos en la configuración por defecto.

    Parámetros
    ----------
    n_policies : int
        Número de pólizas a generar. Por defecto 30,000.
    seed : int
        Semilla de reproducibilidad. Por defecto 20260618 (semilla base del proyecto).
    include_quality_noise : bool
        Extensión futura: introduce nulos y outliers controlados en columnas
        seleccionadas. Por defecto False — 0 nulos en el dataset base
        (validaciones §6.4 del diseño técnico).

    Retorna
    -------
    pd.DataFrame
        DataFrame con shape (n_policies, 21) según el schema del Subtema 4.
        Columnas 1–16: features. Columnas 17–21: targets.
        Los targets nunca se usan como features de clustering (§2.2 del diseño).

    Raises
    ------
    NotImplementedError
        Si include_quality_noise=True (funcionalidad pendiente de task handoff).
    ValueError
        Si n_policies < 1.
    """
    if include_quality_noise:
        raise NotImplementedError(
            "include_quality_noise=True es una extensión futura (§6.1 del diseño). "
            "Por defecto False garantiza 0 nulos y sin outliers artificiales."
        )
    if n_policies < 1:
        raise ValueError(f"n_policies debe ser >= 1. Recibido: {n_policies}")

    rng = np.random.default_rng(seed)
    n = n_policies

    # ── Col 1: Identificador (nunca feature) ─────────────────────────────────
    id_poliza_sin = [f"SIN-{i:06d}" for i in range(1, n + 1)]

    # ── Col 2–3: Variables temporales ─────────────────────────────────────────
    anio_poliza = rng.choice([2021, 2022, 2023, 2024, 2025], size=n)
    mes_emision = rng.integers(1, 13, size=n)  # uniforme 1–12

    # ── Col 4: Edad — Normal(42, 12) clampada [18, 75] ───────────────────────
    edad = np.clip(rng.normal(42.0, 12.0, n).round(), 18, 75).astype(int)

    # ── Col 5: Sexo ───────────────────────────────────────────────────────────
    sexo = rng.choice(["M", "F"], size=n, p=[0.52, 0.48])

    # ── Col 6: Estado civil ───────────────────────────────────────────────────
    estado_civil = rng.choice(
        ["soltero", "casado", "union_libre", "divorciado", "viudo"],
        size=n,
        p=[0.30, 0.42, 0.15, 0.10, 0.03],
    )

    # ── Col 9: Plan — marginal 40/40/20; se genera antes de num_renovaciones
    #           porque num_renovaciones se condiciona en plan (estructura cluster)
    plan = rng.choice(["Basico", "Amplio", "Plus"], size=n, p=[0.40, 0.40, 0.20])
    uso_vehiculo = rng.choice(["Particular", "Carga", "Comercial", "SCA"], size=n, p=[0.30, 0.40, 0.20, 0.10])
    color_vehiculo = rng.choice(["Blanco", "Rojo", "Negro"], size=n, p=[0.40, 0.40, 0.20])
    # ── Col 7: num_renovaciones — Binomial(5, p) condicionado en edad y plan
    #   Correlación positiva edad–renovaciones para la estructura de clusters:
    #   C1 "Nuevos básicos" (edad<30, reno=0), C3 "Premium fiel" (edad 45-65, reno 4-5)
    plan_reno_add = np.where(plan == "Plus", 0.12,
                    np.where(plan == "Basico", -0.07, 0.00))
    p_reno = np.clip(0.25 + 0.006 * (edad - 18) + plan_reno_add, 0.05, 0.85)
    num_renovaciones = np.clip(rng.binomial(5, p_reno), 0, 5)

    # ── Col 8: Región ─────────────────────────────────────────────────────────
    region = rng.choice(
        ["Norte", "Centro", "Sur", "CDMX", "Occidente"],
        size=n,
        p=[0.20, 0.25, 0.15, 0.25, 0.15],
    )

    # ── Col 10: Tipo de vehículo ──────────────────────────────────────────────
    kilometraje = rng.uniform(0, 200000, size=n)

    tipo_vehiculo = rng.choice(
        ["Sedan", "SUV", "Pickup", "Compacto", "Deportivo"],
        size=n,
        p=[0.30, 0.25, 0.20, 0.20, 0.05],
    )
    
    # ── Col 11: Antigüedad del vehículo — triangular decreciente [0, 20] ──────
    #   Sesgo leve hacia valores bajos; sin correlación fuerte con edad para
    #   mantener la separación C1/C4 en el espacio de clustering.
    antig_pesos = np.array([21 - k for k in range(21)], dtype=float)
    antig_pesos /= antig_pesos.sum()
    antiguedad_vehiculo = rng.choice(np.arange(21), size=n, p=antig_pesos)

    # ── Col 12: Suma asegurada — log-normal modulada por plan ─────────────────
    #   Plan superior → vehículo más costoso → SA más alta
    sa_mu = np.where(plan == "Plus", 12.8,
             np.where(plan == "Basico", 12.3, 12.5))
    suma_asegurada = np.round(rng.lognormal(sa_mu, 0.6, n), 2)

    # ── Col 13: Deducible — Normal condicionada en plan, clip [0.05, 0.30] ────
    ded_mu = np.where(plan == "Basico", 0.25,
              np.where(plan == "Amplio", 0.15, 0.10))
    deducible_pct = np.round(np.clip(rng.normal(ded_mu, 0.02, n), 0.05, 0.30), 4)

    # ── Col 14: Forma de pago ─────────────────────────────────────────────────
    forma_pago = rng.choice(
        ["Anual", "Semestral", "Trimestral", "Mensual"],
        size=n,
        p=[0.35, 0.20, 0.20, 0.25],
    )

    # ── Col 15: Canal de venta ────────────────────────────────────────────────
    canal_venta = rng.choice(
        ["Agente", "Directo", "Digital", "Broker"],
        size=n,
        p=[0.45, 0.20, 0.25, 0.10],
    )

    # ── Col 16: nivel_riesgo_asignado (§3.1, calibrado v2) ───────────────────
    #   Derivado de features de suscripción (nunca de siniestros observados).
    #   Bajo: conductor de edad media + vehículo de bajo riesgo (~30% esperado).
    #   Alto: jóvenes/mayores + deportivos + pickups viejos + pickups en Norte (~22%).
    #   Cambios respecto a v1: Bajo usa 2 condiciones AND (vs 4); Norte solo activa
    #   con Pickup (no con SUV) para calibrar Alto en el rango 15–25%.
    es_bajo = (
        (edad >= 30) & (edad <= 50)
        & np.isin(tipo_vehiculo, ["Sedan", "Compacto"])
    )
    es_alto = (
        (edad < 25)
        | (edad > 65)
        | (tipo_vehiculo == "Deportivo")
        | ((tipo_vehiculo == "Pickup") & (antiguedad_vehiculo > 12))
        | ((region == "Norte") & (tipo_vehiculo == "Pickup"))
    )
    nivel_base = np.where(es_alto, 2, np.where(es_bajo, 0, 1))  # 0=Bajo, 1=Medio, 2=Alto

    # ~10% reasignaciones aleatorias entre categorías adyacentes (incertidumbre de suscripción)
    mask_ruido = rng.random(n) < 0.10
    direccion = rng.integers(0, 2, n)  # 0 = hacia menos riesgo, 1 = hacia más
    delta = np.where(direccion == 0, -1, 1)
    nivel_num = np.where(mask_ruido, np.clip(nivel_base + delta, 0, 2), nivel_base)

    _niveles = np.array(["Bajo", "Medio", "Alto"])
    nivel_riesgo_asignado = _niveles[nivel_num]

    # ── Col 17: prima_neta (§3.2) ────────────────────────────────────────────
    #   Estructura aditiva con término cuadrático en edad centrada.
    #   Objetivo pedagógico: OLS detecta la relación; PolynomialFeatures(2) mejora R².
    edad_c = (edad - 42).astype(float)  # centrada en la media

    riesgo_add = np.where(nivel_riesgo_asignado == "Alto", 2_500.0,
                 np.where(nivel_riesgo_asignado == "Bajo", -800.0, 0.0))
    
    vehiculo_add = np.select(
        [tipo_vehiculo == "Sedan", tipo_vehiculo == "SUV",
         tipo_vehiculo == "Pickup", tipo_vehiculo == "Compacto",
         tipo_vehiculo == "Deportivo"],
        [0.0, 800.0, 1_000.0, -300.0, 2_500.0],
        default=0.0,
    )
    vehiculo_uso_add = np.select(
        [uso_vehiculo == "Carga", uso_vehiculo == "Particular",
         uso_vehiculo == "SCA", uso_vehiculo == "Comercial"],
        [2_000.0, 100.0, 1_000.0, 300.0],
        default=0.0,
    )
    vehiculo_color_add = np.select(
        [color_vehiculo == "Negro", color_vehiculo == "Blanco",
         color_vehiculo == "Rojo"],
        [200.0, 100.0,4_000.0],
        default=0.0,
    )
    region_add = np.select(
        [region == "Norte", region == "Centro", region == "Sur",
         region == "CDMX", region == "Occidente"],
        [500.0, 0.0, -300.0, 300.0, -200.0],
        default=0.0,
    )
    plan_prima_add = np.where(plan == "Plus", 6_000.0,
                     np.where(plan == "Amplio", 2_500.0, 0.0))

    prima_base = (
        3_500.0                     # β₀ — intercepto base
        + 0.010 * suma_asegurada    # β₁ — por MXN asegurado (~2,700 en mediana)
        + 25.0 * edad_c             # β₂ — efecto lineal edad centrada
        + 3.0 * edad_c ** 2         # β₃ — U-shape: jóvenes y mayores pagan más
        + plan_prima_add            # β₄/β₅ — plan Amplio/Plus
        + riesgo_add                # β₆ — factor riesgo asignado
        + vehiculo_add              # β₇ — factor tipo de vehículo
        + region_add                # β₈ — factor región
        + rng.normal(0.0, 600.0, n) # ε — ruido N(0, 600²)
        + kilometraje*0.005 # β₉ — factor kilometraje 
        + vehiculo_uso_add # β₁₀ — factor uso del vehículo
        + vehiculo_color_add # β₁₁ — factor color del vehículo
        )
    # Validación §6.4 #5: prima_neta.min() > 0
    prima_neta = np.round(np.clip(prima_base, 500.0, None), 2)

    # ── Col 18: n_siniestros (§3.3) ──────────────────────────────────────────
    #   Poisson condicionada en nivel_riesgo, antigüedad y tipo de vehículo.
    lam_base = np.where(nivel_riesgo_asignado == "Alto", 0.55,
               np.where(nivel_riesgo_asignado == "Bajo", 0.12, 0.28))
    fact_antig = np.where(antiguedad_vehiculo > 12, 1.35,
                 np.where(antiguedad_vehiculo > 5, 1.15, 1.00))
    fact_tipo = np.select(
        [tipo_vehiculo == "Sedan", tipo_vehiculo == "Compacto",
         tipo_vehiculo == "SUV", tipo_vehiculo == "Pickup",
         tipo_vehiculo == "Deportivo"],
        [1.00, 0.95, 1.10, 1.20, 1.40],
        default=1.00,
    )
    lam = lam_base * fact_antig * fact_tipo
    # Validación §6.4 #7: n_siniestros.max() <= 5
    n_siniestros = np.clip(rng.poisson(lam), 0, 5).astype(int)

    # ── Col 19: monto_siniestro (§3.4) ───────────────────────────────────────
    #   Suma de k realizaciones LogNormal(μ=9.1, σ=0.8), k = n_siniestros.
    #   Vectorizado: matriz (n × 5) con máscara de inclusión.
    severidad_matrix = rng.lognormal(9.1, 0.8, (n, 5))
    k_mask = np.arange(5)[np.newaxis, :] < n_siniestros[:, np.newaxis]
    monto_siniestro = np.round((severidad_matrix * k_mask).sum(axis=1), 2)

    # ── Col 20: siniestro_alto (§3.5) — derivación determinística ────────────
    # Validación §6.4 #3: exactamente 1 iff n_siniestros > 0
    siniestro_alto = (n_siniestros > 0).astype(int)

    # ── Col 21: status_poliza (§3.6) ─────────────────────────────────────────
    #   Logit multinomial en dos etapas.
    #   Coeficientes calibrados para lograr ~72% Vigente, ~18% Cancelada, ~10% Vencida.
    #   (Los coeficientes conceptuales del diseño 2.0 × num_renovaciones producirían
    #    prob_vigente > 0.98 por la media de renovaciones; se escalan manteniendo el
    #    mismo ordenamiento de importancia relativa de predictores.)
    logit_v = (
        0.40 * num_renovaciones
        - 0.60 * (forma_pago == "Mensual").astype(float)
        - 0.40 * (nivel_riesgo_asignado == "Alto").astype(float)
        + 0.30 * (canal_venta == "Agente").astype(float)
        + 0.80 * (anio_poliza >= 2024).astype(float)
        - 0.10  # intercepto de calibración (~72% Vigente)
    )
    prob_vigente = 1.0 / (1.0 + np.exp(-logit_v))

    # Cancelada vs Vencida — entre las no-vigentes
    logit_cancel = (
        1.50 * (n_siniestros >= 2).astype(float)
        + 0.80 * (nivel_riesgo_asignado == "Alto").astype(float)
        - 0.80 * (num_renovaciones == 0).astype(float)
        - 0.50 * (anio_poliza <= 2022).astype(float)
        + 0.60  # intercepto de calibración (~64% Cancelada entre no-vigentes)
    )
    p_cancel_no_vig = 1.0 / (1.0 + np.exp(-logit_cancel))

    status_r = rng.random(n)
    status_poliza = np.where(
        status_r < prob_vigente,
        "Vigente",
        np.where(
            status_r < prob_vigente + (1.0 - prob_vigente) * p_cancel_no_vig,
            "Cancelada",
            "Vencida",
        ),
    )

    # ── Ensamblado del DataFrame ──────────────────────────────────────────────
    df = pd.DataFrame(
        {
            "id_poliza_sin": id_poliza_sin,
            "anio_poliza": anio_poliza.astype(int),
            "mes_emision": mes_emision.astype(int),
            "edad": edad,
            "sexo": pd.Categorical(sexo, categories=["M", "F"]),
            "estado_civil": pd.Categorical(
                estado_civil,
                categories=["soltero", "casado", "union_libre", "divorciado", "viudo"],
            ),
            "num_renovaciones": num_renovaciones.astype(int),
            "region": pd.Categorical(
                region,
                categories=["Norte", "Centro", "Sur", "CDMX", "Occidente"],
            ),
            "plan": pd.Categorical(plan, categories=["Basico", "Amplio", "Plus"]),
            "tipo_vehiculo": pd.Categorical(
                tipo_vehiculo,
                categories=["Sedan", "SUV", "Pickup", "Compacto", "Deportivo"],
            ),
            
            "uso_vehiculo": pd.Categorical(
                uso_vehiculo,
                categories=["Particular", "Carga", "Comercial", "SCA"],
            ),
            
            "color_vehiculo": pd.Categorical(
                color_vehiculo,
                categories=["Blanco", "Rojo", "Negro"],
            ),
            "kilometraje": kilometraje,
            "antiguedad_vehiculo": antiguedad_vehiculo.astype(int),
            "suma_asegurada": suma_asegurada,
            "deducible_pct": deducible_pct,
            "forma_pago": pd.Categorical(
                forma_pago,
                categories=["Anual", "Semestral", "Trimestral", "Mensual"],
            ),
            "canal_venta": pd.Categorical(
                canal_venta,
                categories=["Agente", "Directo", "Digital", "Broker"],
            ),
            "nivel_riesgo_asignado": pd.Categorical(
                nivel_riesgo_asignado,
                categories=["Bajo", "Medio", "Alto"],
                ordered=True,
            ),
            "prima_neta": prima_neta,
            "n_siniestros": n_siniestros,
            "monto_siniestro": monto_siniestro,
            "siniestro_alto": siniestro_alto.astype(int),
            "status_poliza": pd.Categorical(
                status_poliza,
                categories=["Vigente", "Cancelada", "Vencida"],
            ),
        }
    )

    return df


def save_portfolio_v2(
    df: pd.DataFrame,
    output_dir: Path = Path(__file__).parent.parent / "data" / "simulated",
    stem: str = "auto_insurance_synthetic_v2",
) -> Path:
    """
    Guarda el DataFrame generado como CSV en data/simulated/.

    Parámetros
    ----------
    df : pd.DataFrame
        Dataset generado por simulate_auto_insurance_portfolio_v2().
    output_dir : Path
        Directorio de salida. Por defecto data/simulated/.
    stem : str
        Nombre base del archivo (sin extensión).

    Retorna
    -------
    Path
        Ruta al archivo CSV guardado.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{stem}.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


if __name__ == "__main__":
    # Ejecución directa: genera el dataset con semilla base y guarda en data/simulated/
    df = simulate_auto_insurance_portfolio_v2(n_policies=30_000, seed=20260618)
    csv_path = save_portfolio_v2(df)
    print(f"Dataset guardado: {csv_path}")
    print(f"Shape: {df.shape}")
    print(f"Nulos: {df.isnull().sum().sum()}")
    print(f"prima_neta.min(): {df['prima_neta'].min():.2f}")
    print(f"n_siniestros.max(): {df['n_siniestros'].max()}")
    pct_riesgo = df["nivel_riesgo_asignado"].value_counts(normalize=True).mul(100).round(1)
    print(f"nivel_riesgo_asignado (%):\n{pct_riesgo.to_string()}")
    pct_status = df["status_poliza"].value_counts(normalize=True).mul(100).round(1)
    print(f"status_poliza (%):\n{pct_status.to_string()}")
    pct_sin = df["siniestro_alto"].mean() * 100
    print(f"siniestro_alto=1: {pct_sin:.1f}%")
    # Guardar Parquet si pyarrow está disponible
    output_dir = Path(__file__).parent.parent / "data" / "simulated"
    parquet_path = output_dir / "auto_insurance_synthetic_v2.parquet"
    try:
        df.to_parquet(parquet_path, index=False)
        print(f"Parquet guardado: {parquet_path}")
    except ImportError:
        print("pyarrow no disponible — solo se guardó CSV (sin Parquet)")
