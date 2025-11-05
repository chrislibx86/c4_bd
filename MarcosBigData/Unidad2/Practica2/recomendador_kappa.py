import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="📚 Recomendador de Libros - Arquitectura Kappa", layout="wide")

# -----------------------------
# 🔹 CARGA DE DATOS INICIALES
# -----------------------------
st.title("📚 Recomendador de Libros con Arquitectura Kappa")
st.caption("Simulación educativa: procesamiento de flujo continuo sin capa batch.")

df_libros = pd.read_csv("libros_amazon.csv")

# Limpiamos columnas útiles
df_libros = df_libros[[
    "CodigoASIN", "Titulo", "Valoracion", "NumeroResenas", "Categorias"
]]

# Extraer solo el número de la valoración ("4.6 out of 5 stars" -> 4.6)
df_libros["Valoracion"] = df_libros["Valoracion"].str.extract(r"(\d+\.\d+)").astype(float)
df_libros["Categorias"] = df_libros["Categorias"].str.replace(r"[\[\]\"']", "", regex=True)

# -----------------------------
# 🔹 CAPA DE STREAMING (Flujo)
# -----------------------------
st.header("⚡ Flujo de Datos en Tiempo Real (Simulación)")

# Simulamos entrada de un nuevo evento
nuevo_titulo = st.text_input("📘 Nuevo libro leído / valorado:")
nueva_valoracion = st.slider("⭐ Valoración del libro", 1.0, 5.0, 4.5, 0.1)
nuevas_categorias = st.text_input("🏷️ Categorías (separadas por comas):", "Books, Literature & Fiction")

if "df_stream" not in st.session_state:
    st.session_state["df_stream"] = df_libros.copy()

# Cuando llega un nuevo "evento"
if st.button("➕ Enviar nuevo evento"):
    nuevo_registro = {
        "CodigoASIN": f"NEW{len(st.session_state['df_stream'])+1}",
        "Titulo": nuevo_titulo,
        "Valoracion": nueva_valoracion,
        "NumeroResenas": np.random.randint(1, 5000),
        "Categorias": nuevas_categorias
    }
    st.session_state["df_stream"] = pd.concat(
        [st.session_state["df_stream"], pd.DataFrame([nuevo_registro])],
        ignore_index=True
    )
    st.success(f"✅ Nuevo libro agregado: {nuevo_titulo}")

# -----------------------------
# 🔹 PROCESAMIENTO CONTINUO
# -----------------------------
st.header("🧠 Procesamiento Continuo (Modelo en Streaming)")

df = st.session_state["df_stream"]

# Vectorizamos las categorías para similitud semántica
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["Categorias"])

# Similaridad entre libros en tiempo real
sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# -----------------------------
# 🔹 CAPA DE SERVICIO
# -----------------------------
st.header("🚀 Recomendaciones en Tiempo Real")

libro_base = st.selectbox("Selecciona un libro base:", df["Titulo"])

if libro_base:
    idx = df.index[df["Titulo"] == libro_base][0]
    similitudes = pd.Series(sim_matrix[idx], index=df["Titulo"]).sort_values(ascending=False)
    recomendaciones = similitudes[1:6]  # top 5 similares

    st.write(f"🔍 Libros similares a **{libro_base}**:")
    st.dataframe(recomendaciones.round(3), use_container_width=True)
    st.bar_chart(recomendaciones)

st.info("💡 Cada nuevo libro o valoración actualiza el modelo en tiempo real, sin necesidad de procesamiento batch.")


