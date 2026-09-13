# Evaluación — Módulo 4 · Tema 2: GLM con Python

**Alumno:** Dereck
**Variable asignada:** `cobertura`  (cobertura)
**Fecha de entrega:** 13/09/2026

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
**Tu respuesta:** No se cumple, ya que, el φ de Person = 1.166 > 1, lo cual indica que la varianza es mayor que la media, además bajo la prueba de Cameron-Trivedi podemos rechazar la **equidispersión** (α = 0.074, p = 3.7e-56 < 0.05). Sin embargo, la sobredispersión es leve (φ < 1.5) por lo que una **QuasiPoisson** que corriga los errores por la √φ es suficiente y podemos mantener los coeficientes de la poisson. Se podria analizar el ajuste a una Binomial Negtiva, pero la sobredispersión no es suficiente para justificar su uso (se recomienda cuando φ > 2 )

**P2.** Interpreta los rating factors de tu variable: nivel más alto y más bajo, traducidos a % de
recargo/descuento. ¿Algún IC cruza 1 o tiene p > 0.05? ¿Qué harías con ese nivel?
**Tu respuesta:** La referencia es la cobertura Amplia (RF=1), se encuentra como el grupo intermedio respecto a riesgo, El RF más alto es de la cobertura RC con 1.079, lo que indica que la frecuencia estimada de siniestro es mayor con un **recargo de 7.9%** respecto a la cobertura Amplia; el más bajo es la cobertura Limitada con 0.9445 un **descuento del 5.55%**. Los asegurados que contratan RC tienen mayor frecuencia de reclamación que aquellos con cobertura Amplia, mientras que los de cobertura Limitada son los que menor frecuencia registran.
El Intervalo de Confianza de RC está por encima de 1 (1.0348-1.1251), como todo el intervalo es mayor a 1, se confirma con un 95% de certeza que este grupo presenta mayor siniestralidad y todos los p < 0.05, por lo que se justifica mantener los tres grupos dentro de la tarificación para segmentar el riesgo.

**P3.** ¿Por qué el GLM one-way reproduce exactamente la tasa empírica, y qué aporta el GLM que una
tabla empírica no puede dar?
**Tu respuesta:**

En un GLM one-way, la tasa ajustada coincide con la tasa empírica ponderada. Esto se debe a que la función de enlace logarítmica y el uso del offset de exposición obligan a que, para cada nivel del factor, la suma de los valores predichos sea igual a la suma de los observados (la pequeña diferencia de 2.4e-05 es solo ruido numérico).

Respecto a una tabla empírica, el GLM aporta la evaluación de significancia estadística a través del p-value (para verificar si cada cobertura es estadísticamente distinta de la referencia) y proporciona intervalos de confianza para cuantificar los márgenes de riesgo. Además, permite integrar múltiples variables explicativas de forma multiplicativa y aislar el efecto individual de cada una.

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

La distribución gamma es adecuada para modelar la severidad porque se usa cuando los valores son positivos, además de que tiene CV constante (la variabilidad relativa del monto es
parecida en todos los niveles), lo que se acopla bien al comportamiento de la severidad en seguros. No se ajusta una regresion lineal sobre log(Y) porque no estarías estimando la media de la severidad, sino su logaritmo (E[log(y)], dificil de interpretar) y al aplicar exponencial para regresar a la escala, se genera un sesgo sistemático, requiriendo un factor de corrección. En cambio, el GLM Gamma con liga logarítmica modela la media esperada en la escala original monetaria. La distribución Lognormal no es un GLM porque no pertenece a la familia exponencial sobre Y (una de las propiedades de los GLM).

**P5.** Según la tabla de comparación, ¿qué modelo elegirías? Justifica con AIC/BIC. ¿Por qué el pseudo R²
es tan bajo y eso NO significa que el modelo sea malo?
**Tu respuesta:**
Elegiría **Binomial Negativa**, ya que tiene menor AIC y BIC respecto al modelo Poisson. El AIC es 124,925.7 vs 125,081.7 del modelo Poisson y el BIC es de 125,105.8 vs 125,261.7.
Al tener ambos criterios con valores más bajos, la Binomial Negativa tiene un mejor ajuste considerando la complejidad del modelo. Esto es consistente con la sobredispersión que detectamos al inicio. Por otro lado, un pseudo R² tan bajo (~0.02) no significa que el modelo sea malo. En modelos de frecuencia de seguros es común obtener valores bajos debido a la alta aleatoriedad que ningún modelo puede explicar. El objetivo del modelo no es explicar toda la varianza individual, sino diferenciar y segmentar los niveles de riesgo entre diferentes asegurados.

**P6.** Compara tus rating factors de frecuencia (Parte 1) con los de severidad para `cobertura`. ¿Apuntan en
la misma dirección? ¿Qué implica eso para separar Frecuencia × Severidad?
**Tu respuesta:** No apuntan en la misma dirección: En Frecuencia, el orden de riesgo es RC > Amplia > Limitada (RFs: 1.0790, 1.0000, 0.9445). RC presenta un recargo respecto a Amplia. En Severidad, el orden de riesgo cambia a Amplia > RC > Limitada (RFs: 1.0000, 0.8533, 0.6681). Aquí RC presenta un descuento del 14.67% respecto a la cobertura Amplia, mientras que Limitada mantiene el mayor descuento (33.19%). Además, los RFs de severidad muestran una variación mucho más amplia (0.6681 a 1.0000) que los de frecuencia (0.9445 a 1.0790), lo que indica que la variable cobertura tiene una mayor capacidad de discriminación sobre el monto del siniestro que en la probabilidad de que ocurra.
Si combináramos ambos componentes en un solo modelo directo de costo puro, los efectos se compensarían parcialmente y ocultarían dinámicas de riesgo distintas: los asegurados con RC chocan con mayor frecuencia, pero sus reclamos son de menor cuantía promedio en comparación con los de cobertura Amplia, es por eso que se modela Frecuencia x Severidad.

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
El ratio pred/obs es de 1.025, lo cual es bastante cercano a 1, lo que indica que el modelo está bien calibrado a nivel global (sobreestima el volumen total de siniestros un 2.5%). El Gini es de 0.23, menor que 0.3, por lo que podemos concluir que el modelo está discriminando de forma moderada o baja (identifica los grupos con menor o mayor riesgo). Las dos métricas tienen un objetivo distinto e importante: El ratio pred/obs mide la calibración global del modelo (qué tan cercanas son las predicciones respecto a los valores observados), y Gini nos indica si puede segmentar de forma adecuada el riesgo (importante para saber si puede diferenciar entre buenos y malos riesgos).

**P8.** Lee la tabla de tarifa: ¿qué nivel de tu variable paga la prima pura más alta y cuál la más baja?
Traduce el factor de tarifa a un recargo/descuento sobre la prima promedio.
**Tu respuesta:** La prima pura más alta es la de la cobertura de RC con $191.77 y un factor de tarifa 1.0518, lo que indica un **recargo de 5.18%** sobre la prima promedio. La más baja es para la cobertura Limitada con $163.39 y un factor 0.8961 lo que significa un **descuento del 10.39%**. Esto es consistente con los resultados anteriores: la cobertura Limitada presenta la prima más baja debido a que registra la menor frecuencia y el menor costo medio por siniestro y la cobertura RC resulta en la prima más alta porque, aunque su severidad es intermedia, presenta la mayor frecuencia de siniestros, lo que termina elevando el costo puro esperado por exposición. 

**P9. (Conclusión de nota técnica).** En 3–4 líneas, redacta cómo `cobertura` afecta la tarifa, integrando
frecuencia, severidad y prima pura, en estilo defendible ante la CNSF.
**Tu respuesta:**

La cobertura que contrata el asegurado es un factor de tarificación indispensable. El efecto se concentra en la severidad: La cobertura Amplia presenta una severidad esperada 1.5 veces mayor que la cobertura Limitada y 1.17 que la de RC, mientras que el efecto sobre la frecuencia es moderado entre niveles. Combinando ambos componentes, la prima pura de la cobertura RC es 1.05 veces la prima promedio, frente a un descuento del 10.39% para la cobertura Limitada. Se recomienda mantener segmentada la cartera con estas tres coberturas por su consistencia y su capacidad de discriminación.

---
*Evaluación generada automáticamente · Diplomado ML en Seguros · FC UNAM · Módulo 4 · Tema 2*
