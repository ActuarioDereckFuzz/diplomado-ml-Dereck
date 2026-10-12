# Tarea 1 — Publica tu tablero

**Diplomado Machine Learning en Seguros · FC UNAM · Módulo 5 · Tema 1**

> **Peso:** 0.5 · **Evalúa:** el Ayudante · **Entrega:** [fecha] por [medio]

---

## Qué debes entregar

Publica en internet el **tablero que construimos en clase** y entrega **dos links**:

1. **El link público de tu tablero** — una URL `https://....streamlit.app` que **abra y funcione**.
2. **El link de tu repositorio de GitHub** (público) de donde se despliega.

El objetivo es **lograr publicar**: que tu tablero deje de vivir solo en tu laptop y quede en una
URL que cualquiera pueda abrir. Es el mismo tablero de clase —no tienes que inventar nada nuevo—;
el reto es el **deploy**.

> Usa la **Guía de autoestudio de deploy** como receta (Partes A y B). La Parte C es el bonus.

---

## Qué debe tener tu tablero (lo de clase)

- **Filtros** en la barra lateral (cobertura, sexo, uso, combustible, edad).
- **Tarjetas de KPIs** (exposición, nº de siniestros, frecuencia, severidad, prima pura).
- Al menos **una gráfica** (frecuencia por factor y/o combined ratio por factor).
- El **cotizador** funcionando, con las tres primas **en orden** (1 empírico → 2 GLM → 3 precio) y
  el cotizador empírico **corregido** (que devuelve prima, no se queda vacío).

No necesitas agregarle nada nuevo. Lo que se evalúa es que **esté publicado y funcione**.

---

## Qué debe tener tu repo (público, nuevo)

- `app_streamlit.py` (en la raíz del repo)
- `datos/datos.parquet`
- `requirements.txt` (con `streamlit, pandas, numpy, matplotlib, statsmodels, pyarrow`)
- `.gitignore` que **excluya `cartera.db`** (la base NO se sube; se reconstruye sola)
- `README.md` con una línea de qué hace y cómo correrlo local

---

## Rúbrica (0.5 en total)

| Criterio | Puntos | Qué se revisa |
|---|---:|---|
| **El link abre y funciona** | **0.20** | La URL `*.streamlit.app` carga el tablero sin errores. **Es el corazón de la tarea.** |
| **Elementos del tablero** | **0.10** | Filtros, tarjetas de KPIs, al menos una gráfica, y el cotizador funcionando (empírico corregido + 3 primas en orden). |
| **Repo bien armado** | **0.10** | `requirements.txt` completo (con `pyarrow`), el parquet en `datos/`, y **`cartera.db` NO subida**. |
| **README** | **0.05** | Describe el tablero y dice cómo correrlo local. |
| **El repo despliega el link** | **0.05** | El link sale del repo entregado (un `push` lo actualiza). |

> Media tarea se la lleva que el link **abra y funcione**. Si no logras el deploy permanente pero
> entregas un link de **VS Code port forwarding activo al momento de revisar**, cuenta como entrega
> **parcial** del criterio del link (la mitad).

---

## ⭐ Bonus (extra) — Que tu tablero RECUERDE: hasta +0.10

Opcional. Súmale memoria a tu tablero: que **cada cotización se guarde** en una **Google Sheet**
(ver Parte C de la guía). Se evalúa así:

| Bonus | Puntos extra | Qué se revisa |
|---|---:|---|
| **Persiste una tarificación** | **+0.10** | Cotizas en tu app publicada, la fila **aparece** en tu Google Sheet y **sigue ahí** tras reiniciar. Credenciales vía `st.secrets` (no en el código). |

> El bonus **suma** sobre tu calificación; no es requisito. Es para quien quiera ir un paso más
> allá y ver cómo un tablero registra su propio uso.

---

## Errores típicos (evítalos)

- Olvidar **`pyarrow`** en `requirements.txt` → el deploy truena al leer el parquet.
- No subir el **parquet** a `datos/` → la app no encuentra los datos en el servidor.
- Subir **`cartera.db`** por error → va en `.gitignore`.
- Repo **privado** → para Community Cloud gratis, hazlo **público**.
- Entregar solo el repo **sin el link**, o el link **sin el repo** → se piden **los dos**.
- (Bonus) Compartir la Google Sheet como **Viewer** → no deja escribir; debe ser **Editor**.

---

## Checklist antes de entregar

- [ ] Mi link `https://....streamlit.app` abre y veo el tablero funcionando.
- [ ] Puedo usar los filtros y el cotizador en la versión publicada.
- [ ] Mi repo es **público** y tiene los 5 elementos de arriba.
- [ ] En GitHub **no** aparece `cartera.db`.
- [ ] Entrego **los dos links**: tablero + repo.
- [ ] (Bonus) Al cotizar, la fila queda en mi Google Sheet.
