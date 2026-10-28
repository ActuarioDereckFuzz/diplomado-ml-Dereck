# Reto del sábado — Paquete de cierre mensual

**Diplomado Machine Learning en Seguros · FC UNAM · Módulo 5 · Tema 2 · Parte 3**

> Práctica guiada + entrega. Construyes, siguiendo al profesor y completando el scaffold, **un
> solo archivo de Excel** que junta el trabajo de la semana: tarifa, cotizador, reservas, resumen
> y control. Puede usarse como **autoevaluación** (checklist) o como **punto extra de la evaluación de la siugiente semana**

---

## Qué entregas

Un archivo **`paquete_cierre.xlsx`** generado por **tu script/notebook** (a partir de
`scaffold_paquete.ipynb`), llenando el master template `plantilla_paquete.xlsx` con **openpyxl**.
Entregas: el **.xlsx** y el **notebook** que lo genera.

El cálculo (GLM Tweedie y chain-ladder) ya viene dado; lo que se evalúa es el **export y el
ensamblado**.

---

## Qué debe tener el paquete (5 pestañas)

1. **Portada** — fecha (`=TODAY()`) y tu nombre.
2. **Tarifa** — prima base, las 6 tablas de factores, y el **cotizador vivo** (dropdowns + VLOOKUP;
   la prima se recalcula al elegir).
3. **Reservas** — triángulo, factores de desarrollo, e **IBNR como fórmulas vivas** (CDF = producto
   de las celdas de factores), con total y comparativo **CL vs BF**.
4. **Resumen** — prima base e IBNR totales, por **referencia** a las otras pestañas.
5. **Control** — reconciliación **Python vs Excel** de pricing y de reservas: ambas diferencias ≈ 0.

---

## Rúbrica (checklist / calificación)

| Criterio | Puntos | Qué se revisa |
|---|---:|---|
| **El archivo abre y tiene las 5 pestañas** | **0.20** | Se generó desde el notebook, sin errores. |
| **Cotizador vivo funciona** | **0.20** | Dropdowns + VLOOKUP; la prima cambia al elegir perfil. |
| **IBNR con fórmulas vivas** | **0.20** | El IBNR sale de fórmulas (no números pegados) y el total cuadra. |
| **Control reconcilia (≈ 0)** | **0.25** | Pricing y reservas: Excel = Python. **El sello de calidad.** |
| **Formato y orden** | **0.15** | Moneda/decimales correctos, factores con precisión completa, pestañas limpias. |

**Bonus (+0.10):** un *plus* a elección — proteger la hoja Tarifa dejando solo los inputs
editables, **o** agregar una gráfica, **o** demostrar la reproducibilidad (re-generar idéntico).

---

## Pistas rápidas

- `ws.cell(fila, col, valor)` escribe; `ws.cell(...).number_format = "#,##0"` da formato.
- Las fórmulas van como texto en **inglés**: `ws.cell(r, c, "=B2*PRODUCT(F5:F10)")`.
- Guarda el **factor con precisión completa**; el formato solo lo *muestra* a 3 decimales (si lo
  redondeas, el Control deja de cuadrar).
- Llena **primero** el triángulo y los factores, y **después** la tabla de IBNR (sus fórmulas
  dependen de esas celdas).

## Checklist antes de entregar

- [X ] Las 5 pestañas están llenas.
- [X ] El cotizador cambia la prima al elegir en los menús.
- [X ] El IBNR total sale de `=SUM(...)` y el comparativo CL vs BF está.
- [X ] En Control, las dos diferencias son ≈ 0.
- [X ] Entrego el `.xlsx` **y** el notebook.
