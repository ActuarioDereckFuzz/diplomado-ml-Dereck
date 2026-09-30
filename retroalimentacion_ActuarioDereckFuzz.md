# Retroalimentación — Dereck
**Repo:** https://github.com/ActuarioDereckFuzz/diplomado-ml-Dereck
**Fecha de evaluación:** 2026-05-12
**Calificación final:** 10.0 / 10

## Resumen general
Excelente entrega.

## Desglose por sesión

### Sesión 1 — Instalación y configuración de Python — 1/1 pt
Archivo: `diplomado-ml-Dereck\Modulo1\Sesion1\sesion1_M1_notebook.ipynb`. 32 de 32 celdas de código con contenido ejecutable.

### Sesión 2 — Tipos de datos básicos — 1/1 pt
Archivo: `diplomado-ml-Dereck\Modulo1\Sesion2\sesion2_M1_notebook.ipynb`. 30 de 30 celdas de código con contenido ejecutable.

### Sesión 3 — Estructuras de datos + control de flujo — 1/1 pt
Archivo: `diplomado-ml-Dereck\Modulo1\Sesion3\sesion3_M1_notebook.ipynb`. 36 de 36 celdas de código con contenido ejecutable.

### Sesión 4 — Funciones — 1/1 pt
Archivo: `diplomado-ml-Dereck\Modulo1\Sesion4\sesion4_M1_notebook.ipynb`. 22 de 22 celdas de código con contenido ejecutable.

### Sesión 5 — Módulos / funciones avanzadas — 2.00/2 pts
Archivo: `diplomado-ml-Dereck\Modulo1\sesion5\sesion5_M1_notebook.ipynb`.
- Notebook ejecutable: **0.5/0.5**
- Ejercicios completos: **0.5/0.5**
- Resultados correctos: **1.0/1.0**
- Las 3 tareas integradoras tienen código.
- Notebook ejecuta end-to-end sin errores.

### Sesión 6 — Módulos propios + inicio de Pandas — 2.0/2 pts
Archivo: `diplomado-ml-Dereck\Modulo1\sesion6\sesion6_M1_notebook.ipynb`.
- Notebook ejecutable: **0.5/0.5**
- Ejercicios completos: **0.5/0.5**
- Resultados correctos: **1.0/1.0**  _(re-evaluado con comparación numérica estricta contra canónico)_
  - Tarea 1: ✓ correcta (inferida — la tarea final integradora produce el resultado esperado)
  - Tarea 2: ✓ correcta (25/25 números coinciden)
  - Tarea 3: ✓ correcta (13/14 números coinciden)

### Sesión 7 — Pandas avanzado — 2.0/2 pts
Archivo: `diplomado-ml-Dereck\Modulo1\sesion7\sesion7_M1_notebook.ipynb`.
- Notebook ejecutable: **0.5/0.5**
- Ejercicios completos: **0.5/0.5**
- Resultados correctos: **1.0/1.0**  _(re-evaluado con comparación numérica estricta contra canónico)_
  - Tarea 1: ✓ correcta (13/15 números coinciden)
  - Tarea 2: ✓ correcta (18/18 números coinciden)
  - Tarea 3: ✓ correcta (30/30 números coinciden)
  - Tarea 4: ✓ correcta (15/15 números coinciden)


## Parte 2 — Sesiones 8 y 11

### Sesión 8 — Pipeline Completo — 1.23/2 pts
*(solo se evaluó el Ejercicio Integrador Final)*
- Corre hasta el Integrador: 0.25/0.5
  - Errores en celdas antes del integrador: [35]
- Integrador Final, 5 fases: 0.98/1.5
  - FASE 1: 0.00*0.3 = 0.00 (celda vacia o solo placeholder)
  - FASE 2: 0.70*0.3 = 0.21 (codigo extenso (31 lineas) ejecuta; printeos no coinciden con canonico)
  - FASE 3: 1.00*0.3 = 0.30 (numeros coinciden (100% de canon cubierto))
  - FASE 4: 0.90*0.3 = 0.27 (mayoria de numeros coinciden (42%))
  - FASE 5: 0.65*0.3 = 0.20 (pocas coincidencias numericas (11%); codigo sustantivo y ejecuta)

### Sesión 11 — Práctica Final — 7.12/8 pts
*(solo se evaluó el Ejercicio Integrador Final "Mini Pricing Actuarial")*
- Corre hasta el Integrador: 1.00/1.0
  - FASE 1: 1.00*1.75 = 1.75 (numeros coinciden (66% de canon cubierto))
  - FASE 2: 1.00*1.75 = 1.75 (numeros coinciden (56% de canon cubierto))
  - FASE 3: 0.80*1.75 = 1.40 (coinciden algunos numeros clave (18%))
  - FASE 4: 0.70*1.75 = 1.22 (codigo extenso (107 lineas) ejecuta; printeos no coinciden con canonico)

## Bonus — Sesiones 9 y 10
- S9 entregada correctamente: **Sí** (integrador correcto (100% de numeros coinciden))
- S10 entregada correctamente: **Sí** (mpl_ok=True sb_ok=True)
- Bonus aplicado: **+0.5**

## Calificación final
- Parte 1: 10.00 / 10
- Parte 2: 8.35 / 10
- Promedio: 9.175 / 10
- Bonus: +0.5
- **Final: 9.68 / 10**

---

# Retroalimentación — Módulo 4 · Tema 2 (GLM con Python)

**Alumno:** Dereck Favila Limón
**Variable asignada:** `cobertura`

## Desglose por pregunta

| Pregunta | Pts | Comentario |
|---|---|---|
| P1 | 9/10 | Diagnóstico correcto y completo: φ=1.166>1, Cameron-Trivedi rechaza equidispersión (p=3.7e-56), identificas correctamente que la sobredispersión es leve (φ<1.5) y que por eso QuasiPoisson basta, sin necesidad de Binomial Negativa (φ>2). Solo hay erratas menores (“Person”, “corriga”). |
| P2 | 9/10 | Interpretación sólida: RC +7.9% recargo, Limitada -5.55% descuento, correctamente identificas que ningún IC cruza 1 y ambos p<0.05, justificando mantener los tres niveles segmentados. |
| P3 | 9/10 | Explicas bien el mecanismo (offset + liga log fuerza suma predicha = suma observada) y lo que aporta el GLM (p-values, IC, combinación multiplicativa de variables). |
| P4 | 9/10 | Buena explicación de CV constante en Gamma, del problema de modelar E[log Y] en vez de E[Y] con el sesgo de retransformación, y de por qué Lognormal no pertenece a la familia exponencial. |
| P5 | 9/10 | Elección correcta de Binomial Negativa citando tus propios AIC/BIC, y explicas adecuadamente por qué un pseudo R² bajo es normal en frecuencia de siniestros. |
| P6 | 10/10 | Excelente: detectas que el orden de riesgo cambia entre frecuencia (RC>Amplia>Limitada) y severidad (Amplia>RC>Limitada), cuantificas los descuentos de severidad y conectas bien esto con la necesidad de modelar Frecuencia × Severidad por separado. |
| P7 | 9/10 | Distingues correctamente calibración (ratio 1.025, sobreestimación de 2.5%) de discriminación (Gini 0.23, modesta), con buena explicación de qué mide cada métrica. |
| P8 | 9/10 | Lectura correcta de la tabla: RC prima más alta ($191.77, +5.18%), Limitada la más baja ($163.39, -10.39%), con narrativa consistente con frecuencia y severidad. |
| P9 | 8/10 | Buena nota técnica que integra frecuencia, severidad y prima pura con números correctos (Amplia 1.5x severidad de Limitada, 1.17x de RC). Le faltó incluir explícitamente un intervalo de confianza para reforzar el argumento defendible ante CNSF. |

## Redacción: 9/10

## Nota final: 90/100 (calificación: 9.0/10)

## Comentarios generales
Evaluación muy sólida y consistente en las tres partes, con buen manejo tanto conceptual (equidispersión, familia Gamma, calibración vs. discriminación) como numérico (citas tus propios rating factors y factores de tarifa con precisión). Destaca especialmente tu respuesta en P6, donde detectas bien que frecuencia y severidad apuntan en direcciones distintas para `cobertura`. Para subir un poco más, procura incluir intervalos de confianza en la nota técnica final (P9) para que el argumento sea aún más defendible ante el regulador. Muy buen trabajo.
