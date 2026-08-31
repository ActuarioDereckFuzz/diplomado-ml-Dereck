# Evaluación — Módulo 4 · Tema 2: GLM con Python

**Alumno:** Dereck
**Variable asignada:** `cobertura`  (cobertura)
**Fecha de entrega:** ________________

> **Instrucciones.** Este archivo evalúa las tres sesiones del tema. Las tablas ya vienen
> calculadas; tu trabajo es **responder las preguntas de interpretación** en el espacio
> "**Tu respuesta:**". Se evalúa la interpretación, no el código. Máx. 4–6 líneas por respuesta.
> Todas tus preguntas usan **tu variable asignada** (`cobertura`). Guarda y sube este archivo a tu repositorio.

---

## Parte 1 · Sesión 1 — Modelo de Frecuencia

**Diagnóstico del supuesto de Poisson (modelo completo):**

| métrica | valor |
| --- | --- |
| φ de Pearson | 1.1664 |
| Cameron-Trivedi α | 0.0744 |
| z | 15.80 |
| p-value | 3.7e-56 |

**Rating factors de frecuencia para `cobertura`** (base × RF reproduce la tasa empírica; diferencia máx = 2.4e-05):

| nivel | RF_frec | IC_inf | IC_sup | p | tasa_emp |
| --- | --- | --- | --- | --- | --- |
| Amplia (ref) | 1 | 1 | 1 | 0 | 0.1352 |
| Limitada | 0.9445 | 0.9017 | 0.9895 | 0.0161 | 0.1277 |
| RC | 1.079 | 1.0348 | 1.1251 | 0.0004 | 0.1459 |

**P1.** ¿Se cumple la equidispersión? Justifica con φ **y** con Cameron-Trivedi, y di qué familia usarías.
**Tu respuesta:**

**P2.** Interpreta los rating factors de tu variable: nivel más alto y más bajo, traducidos a % de
recargo/descuento. ¿Algún IC cruza 1 o tiene p > 0.05? ¿Qué harías con ese nivel?
**Tu respuesta:**

**P3.** ¿Por qué el GLM one-way reproduce exactamente la tasa empírica, y qué aporta el GLM que una
tabla empírica no puede dar?
**Tu respuesta:**

---

## Parte 2 · Sesión 2 — Severidad y Selección de Modelos

**Comparación de modelos de frecuencia:**

| modelo | AIC | BIC | pseudoR2_McF |
| --- | --- | --- | --- |
| Poisson | 125,081.7 | 125,261.7 | 0.0198 |
| Binomial Negativa | 124,925.7 | 125,105.8 | 0.021 |

**Rating factors de severidad (Gamma) para `cobertura`:**

| nivel | RF_sev | severidad_emp |
| --- | --- | --- |
| Amplia (ref) | 1 | 1,589 |
| Limitada | 0.6681 | 1,061 |
| RC | 0.8533 | 1,356 |

**P4.** ¿Por qué se usa **Gamma** para severidad y no una regresión lineal sobre log(Y)? (menciona la
propiedad del CV y por qué Lognormal no es GLM).
**Tu respuesta:**

**P5.** Según la tabla de comparación, ¿qué modelo elegirías? Justifica con AIC/BIC. ¿Por qué el pseudo R²
es tan bajo y eso NO significa que el modelo sea malo?
**Tu respuesta:**

**P6.** Compara tus rating factors de frecuencia (Parte 1) con los de severidad para `cobertura`. ¿Apuntan en
la misma dirección? ¿Qué implica eso para separar Frecuencia × Severidad?
**Tu respuesta:**

---

## Parte 3 · Sesión 3 — Validación y Tarifa

**Validación out-of-sample del modelo de frecuencia:**

| metrica | valor | ideal |
| --- | --- | --- |
| Gini (test) | 0.2315 | > 0.30 aceptable |
| Ratio pred/obs (test) | 1.0249 | ≈ 1.00 |

**Prima pura por nivel de `cobertura`** (Frecuencia × Severidad, con su factor de tarifa):

| nivel | prima_pura_modelo | factor_tarifa |
| --- | --- | --- |
| Amplia (ref) | 182.29 | 0.9997 |
| Limitada | 163.39 | 0.8961 |
| RC | 191.77 | 1.0518 |

**P7.** Interpreta las métricas de validación: ¿el modelo está bien calibrado (ratio pred/obs)? ¿discrimina
bien el riesgo (Gini)? ¿Qué mide cada una?
**Tu respuesta:**

**P8.** Lee la tabla de tarifa: ¿qué nivel de tu variable paga la prima pura más alta y cuál la más baja?
Traduce el factor de tarifa a un recargo/descuento sobre la prima promedio.
**Tu respuesta:**

**P9. (Conclusión de nota técnica).** En 3–4 líneas, redacta cómo `cobertura` afecta la tarifa, integrando
frecuencia, severidad y prima pura, en estilo defendible ante la CNSF.
**Tu respuesta:**

---
*Evaluación generada automáticamente · Diplomado ML en Seguros · FC UNAM · Módulo 4 · Tema 2*
