#%%
import os
import pandas as pd


BANDAS_EDAD = [17, 30, 35, 45, 50, 55, 60, 95]
candidatos = [
        "datos/datos.parquet", "datos/datos.pkl",
        "../../Modulo_4/m4t2_sesion1/datos/datos.parquet",
        "../../Modulo_4/m4t2_sesion1/datos/datos.pkl",
    ]
ruta = next((r for r in candidatos if os.path.exists(r)), None)
df = pd.read_parquet(ruta) if ruta.endswith(".parquet") else pd.read_pickle(ruta)
df["edad_cat"] = pd.cut(df["edad_conductor"], bins=BANDAS_EDAD)


# %%

BANDINGS = {"edad_conductor": [17, 30, 35, 45, 50, 55, 60, 95],
             "antiguedad": [-1,1,2, 3,4, 5, 10, 15, 50],
             "potencia": [9,40,50,60,70,250],
             "nivel_bonus": [-1,0,1,2,5,10,25]}
   
def banda(col, val):
      return str(pd.cut([val], bins = BANDINGS[col])[0])