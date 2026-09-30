
#%%
import pandas as pd
ruta = r"C:\Users\Eric_Daniel\Documents\Ciencias Cursos\Diplomado\diplomado-ml-seguros\diplomado-ml-seguros\Modulo_4\m4t2_sesion1\datos"
pd.read_pickle(ruta + r"\datos.pkl").to_parquet(ruta + r"\datos.parquet")
print("listo, parquet creado")

# %%
