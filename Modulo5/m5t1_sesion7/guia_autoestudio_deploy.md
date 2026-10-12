# Guía de autoestudio — Publica tu tablero en internet

**Diplomado Machine Learning en Seguros · FC UNAM · Módulo 5 · Tema 1 · Sesión 6**

> Esta guía reemplaza la clase del sábado. La haces **solo**, a tu ritmo. Al terminar, tu tablero
> vive en una URL pública que puedes compartir, y —si haces el extra— **recuerda** cada cotización.
>
> **Meta:** un link `https://....streamlit.app` que abre tu tablero sin errores.
>
> **Necesitas:** tu carpeta del tablero (con `app_streamlit.py` corriendo local), una cuenta de
> **GitHub** (gratis) y **Visual Studio Code**.

---

## Panorama

1. **Verlo rápido (VS Code)** — en 2 minutos, pero solo vive mientras tu laptop está prendida.
2. **Publicarlo permanente (Streamlit Community Cloud)** — vive en internet solo. **Esto es lo que
   entregas en la Tarea.**
3. **(Extra) Que recuerde (Google Sheets)** — que cada cotización se guarde. Es el **bonus** de la
   Tarea.

---

## Parte A — Link rápido con VS Code (opcional, 2 min)

1. Corre tu tablero: `streamlit run app_streamlit.py` → queda en el puerto **8501**.
2. En VS Code abre el panel **PORTS** (junto a la TERMINAL; o *View → Open View… → Ports*).
3. Clic en **"Forward a Port"** → escribe **8501**.
4. Si te lo pide, **inicia sesión con GitHub**.
5. Verás una **"Forwarded Address"** en **Private**. Clic derecho sobre el puerto →
   **Port Visibility → Public**.
6. Copia esa dirección y ábrela. ¡Ese es tu tablero en internet!

> ⚠️ Si cierras VS Code o apagas la laptop, **ese link se muere**. Para algo permanente, Parte B.

---

## Parte B — Deploy permanente en Streamlit Community Cloud

**La idea:** Community Cloud publica tu app, pero **no sube tu carpeta directo: la lee desde un
repositorio de GitHub.** Por eso primero llevamos la app a GitHub, en un **repo público nuevo**
(tu repo del diplomado es privado, y el plan gratis trabaja con públicos).

### Paso 1 · Arma la carpeta del tablero

Deja **exactamente** estos archivos:
```
tablero-seguros/
   app_streamlit.py
   requirements.txt
   .gitignore
   datos/
      datos.parquet
```

**`requirements.txt`** (las librerías que el servidor instalará):
```
streamlit
pandas
numpy
matplotlib
statsmodels
pyarrow
```
> No olvides **`pyarrow`**: es la que lee el `parquet`. Es el error #1 del deploy.

**`.gitignore`** (lo que NO se sube):
```
cartera.db
__pycache__/
*.pyc
.venv/
.streamlit/secrets.toml
```
> **¿Por qué `cartera.db` no se sube?** Porque pesa mucho y tu app la **reconstruye sola** desde el
> parquet la primera vez (el `if not os.path.exists(...)` de `cargar_datos`). Por eso el **parquet
> sí se sube** y la base de datos no.

### Paso 2 · Revisa que las rutas estén ancladas

Al inicio de `app_streamlit.py` debe estar:
```python
AQUI    = os.path.dirname(os.path.abspath(__file__))
RUTA_DB = os.path.join(AQUI, "cartera.db")
# y el parquet también relativo a AQUI
```
**¿Por qué importa?** En el servidor el "directorio de trabajo" no es la carpeta de tu app, es la
**raíz del repo**. Si usas `sqlite3.connect("cartera.db")` a secas, no encuentra los datos. Al
anclar las rutas a `AQUI` (la carpeta del propio archivo), la app halla todo **igual en tu laptop
que en el servidor**.

### Paso 3 · Crea el repo público y sube

1. En **github.com** → botón **"New"** → nombre `tablero-seguros` → visibilidad **Public** →
   **Create repository**.
2. Desde tu carpeta, en la terminal:
   ```
   git init
   git add .
   git commit -m "tablero listo para deploy"
   git branch -M main
   git remote add origin https://github.com/TU-USUARIO/tablero-seguros.git
   git push -u origin main
   ```
3. En github.com confirma que están `app_streamlit.py`, `requirements.txt`, `.gitignore`,
   `datos/datos.parquet` — y que **NO** aparece `cartera.db`.

### Paso 4 · Publica en Community Cloud

1. Entra a **https://share.streamlit.io** → inicia sesión con **GitHub** (autoriza la 1ª vez).
2. Clic en **"Create app"** → **"Yup, I have an app."**
3. Llena:
   - **Repository:** `TU-USUARIO/tablero-seguros`
   - **Branch:** `main`
   - **Main file path:** `app_streamlit.py`
4. (Opcional) **"Advanced settings" → Python version: 3.12**. **Save.**
5. Clic en **Deploy** 🚀. La **primera vez tarda unos minutos** (instala las librerías y reconstruye
   la base desde el parquet).
6. Cuando esté lista, copia tu link `https://....streamlit.app`. **Ese es tu entregable.**

> Cada `git push` vuelve a desplegar la app solo.

---

## Parte C (Extra / bonus) — Que tu tablero RECUERDE las cotizaciones

**El problema:** si guardas las cotizaciones en la `cartera.db` del servidor, se borran cuando la
app se duerme o haces un push — **el disco del servidor es efímero**. Para que recuerde de verdad,
necesitas una base **externa**. Aquí: una **Google Sheet**, donde cada cotización es una fila nueva.

### C.1 · Prepara Google (es la parte más tardada; hazla con calma)

1. Crea una **Google Sheet** (ej. `cotizaciones`) y pon encabezados en la fila 1:
   `fecha, edad, cobertura, sexo, uso, combustible, antiguedad, potencia, bonus, prima_glm, tarifa`.
2. Ve a **Google Cloud Console** (console.cloud.google.com):
   - Crea un **proyecto** nuevo.
   - En *APIs y servicios → Biblioteca*, **habilita** *Google Sheets API* y *Google Drive API*.
   - En *APIs y servicios → Credenciales*, crea una **Cuenta de servicio** (Service Account).
   - En esa cuenta, pestaña *Claves* → *Agregar clave* → **JSON** → descarga el archivo.
3. Abre el JSON y copia el **`client_email`** (algo como `...@....iam.gserviceaccount.com`).
   **Comparte tu Google Sheet** con ese correo, con permiso de **Editor** (vamos a escribir, no
   solo leer).

### C.2 · Guarda las credenciales (local)

Crea `.streamlit/secrets.toml` (ya está en `.gitignore`, **no se sube**) y pega el contenido del
JSON así:
```toml
sheet_url = "https://docs.google.com/spreadsheets/d/XXXXXXXX/edit"

[gcp_service_account]
type = "service_account"
project_id = "..."
private_key_id = "..."
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "...@....iam.gserviceaccount.com"
client_id = "..."
token_uri = "https://oauth2.googleapis.com/token"
```
> Regla de oro: **ninguna credencial en el código ni en el repo.** En local viven aquí; en el
> servidor se pegan en el panel de *Secrets* (Paso C.4).

### C.3 · El código: guardar al cotizar

Agrega **`gspread`** a tu `requirements.txt`. Al inicio de `app_streamlit.py`:
```python
import gspread

@st.cache_resource
def hoja():
    gc = gspread.service_account_from_dict(st.secrets["gcp_service_account"])
    return gc.open_by_url(st.secrets["sheet_url"]).sheet1
```
Dentro del Cotizador, cuando el usuario cotiza (después de calcular `pp_glm` y `tarifa`):
```python
    if st.button("Cotizar", type="primary"):
        st.session_state["perfil"] = perfil
        # ... cálculo de pp_glm y tarifa ...
        hoja().append_row([pd.Timestamp.now().isoformat(),
                           perfil["edad_conductor"], perfil["cobertura"], perfil["sexo"],
                           perfil["uso"], perfil["combustible"], perfil["antiguedad"],
                           perfil["potencia"], perfil["bonus"], round(pp_glm, 2), round(tarifa, 2)])
        st.success("Cotización guardada ✔")
```
Para **ver** lo guardado, opcionalmente:
```python
    st.dataframe(pd.DataFrame(hoja().get_all_records()).tail(10))
```

### C.4 · En el servidor

En Community Cloud: *App settings → Secrets* → pega el **mismo** contenido de tu `secrets.toml`.
Redeploy. Cotiza en la app publicada y mira cómo la fila aparece en tu Google Sheet — y sigue ahí
aunque reinicies.

---

## Si algo sale mal — checklist de rescate

| Síntoma | Causa probable | Arreglo |
|---|---|---|
| `ModuleNotFoundError: X` | falta `X` en `requirements.txt` | agrégalo y `git push` |
| `Missing optional dependency 'pyarrow'` | falta `pyarrow` | agrégalo a `requirements.txt` |
| No encuentra el parquet / `no such table` | el parquet no está en `datos/`, o la ruta no está anclada | sube `datos/datos.parquet`; usa `AQUI` |
| El deploy intenta subir ~14 MB | `cartera.db` no está ignorada | agrégala a `.gitignore` |
| 1ª carga lenta | reconstruye la base + ajusta GLM | normal, solo la primera vez |
| "over its resource limits" | pasaste ~1 GB de RAM | reinicia la app desde su panel |
| "This app has gone to sleep" | nadie la visitó | clic en **"Yes, get this app back up!"** |
| (Sheets) error de permiso al guardar | compartiste la hoja como **Viewer** | compártela como **Editor** |
| (Sheets) `SpreadsheetNotFound` | no compartiste la hoja con el `client_email` | compártela con ese correo |
| Warning amarillo de pandas/SQLAlchemy | es solo advertencia, **no** error | ignóralo |

---

## Lo que debes lograr

**Mínimo (Tarea):** un link `https://....streamlit.app` que abre tu tablero sin errores, conectado
a tu repo público de GitHub.

**Bonus (Tarea):** que al cotizar, la fila quede guardada en tu Google Sheet y siga ahí después de
reiniciar.
