import pandas as pd

df_libros = pd.read_csv('Amazon_popular_books_dataset.csv')
df_libros = df_libros.dropna(axis=1)

print(df_libros.columns.to_list())

columnas_es = [
    "CodigoASIN",
    "PreguntasRespondidas",
    "Moneda",
    "EntregaEnvio",
    "Dominio",
    "Caracteristicas",
    "CantidadImagenes",
    "Valoracion",
    "NumeroResenas",
    "NombreVendedor",
    "FechaRegistro",
    "Titulo",
    "EnlaceURL",
    "CantidadVideos",
    "Categorias"
]

df_libros.columns = columnas_es

df_libros.to_csv("libros_amazon.csv", index=False, encoding="utf-8")