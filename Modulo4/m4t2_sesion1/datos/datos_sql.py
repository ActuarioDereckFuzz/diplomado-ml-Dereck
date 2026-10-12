import os
import sqlite3
import pandas as pd

def ruta_parquet():
    rutas = [
        "datos.parquet",
        "../../Modulo4/m4t2_sesion1/datos/datos.parquet",
        "./diplomado-ml-Dereck/Modulo4/m4t2_sesion1/datos/datos.parquet"
    ]
    for r in rutas:
        if os.path.exists(r):
            print(f"✅ Archivo encontrado en: {r}")
            return r
    raise FileNotFoundError("❌ No se encontró el archivo de datos en las rutas especificadas")

# Verificamos el directorio actual
print(f"Directorio de trabajo: {os.getcwd()}")

# Leemos el archivo Parquet (requiere pyarrow o fastparquet instalado)
df = pd.read_parquet(ruta_parquet(), engine="pyarrow")

# Conexión a la base de datos SQLite (crea cartera.db si no existe)
con = sqlite3.connect("cartera.db")

# Guardamos el DataFrame en la tabla 'cartera'
df.to_sql("cartera", con, if_exists="replace", index=False)
print(f"✅ DataFrame guardado en la base de datos 'cartera.db' con {len(df):,} filas")

# # Validamos que se haya insertado correctamente
# tablas = con.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
# print("📋 Tablas disponibles en la BD:", tablas)

# preview = pd.read_sql("SELECT * FROM cartera LIMIT 5", con)
# print("👀 Vista previa de los datos guardados:")
# print(preview)

# Cerramos la conexión
con.close()



# import os, sqlite3
# import pandas as pd
# def ruta_parquet():
#     for r in ["datos/datos.parquet",
#               "../../Modulo_4/m4t2_sesion1/datos/datos.parquet"]:
#         if os.path.exists(r):
#             return r
#     raise FileNotFoundError("No encuentro datos.parquet")
# df  = pd.read_parquet(ruta_parquet())
# con = sqlite3.connect("cartera.db")                 # crea el archivo de la BD
# df.to_sql("cartera", con, if_exists="replace", index=False)   # vuelca el DataFrame a una tabla
# con.close()
# print(f"cartera.db creada: {len(df):,} filas en la tabla 'cartera'")