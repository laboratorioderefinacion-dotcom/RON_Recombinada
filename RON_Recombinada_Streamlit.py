#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ==========================================================
# APP STREAMLIT – RON RECOMBINADA (MODO PRO)
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import warnings

warnings.filterwarnings("ignore")

# ==========================================================
# CONFIGURACIÓN
# ==========================================================

st.set_page_config(
    page_title="RON Recombinada",
    page_icon="🧪",
    layout="centered"
)

# UI limpia (sacar link y padding)
st.markdown("""
<style>
a[href^="#"] { display: none !important; }
.block-container { padding-top: 2rem; }
.big-font { font-size:22px !important; font-weight:bold; }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# HEADER
# ==========================================================

st.markdown("## 🧪 Estimación de RON - Recombinada")

# ==========================================================
# CRITERIOS
# ==========================================================

REPRO_METODO = 0.83
UMBRAL_METODO = REPRO_METODO / 2
UMBRAL_METODO_SUP = 0.6

# ==========================================================
# MODELO
# ==========================================================

@st.cache_resource
def cargar_modelo():
    modelo = load("Modelo_RON_Recombinada.joblib")
    columnas = load("Columnas_RON_Recombinada.joblib")
    return modelo, columnas

try:
    RF, columnas_modelo = cargar_modelo()
    st.success("✅ Modelo Random Forest con validación metrológica")
except:
    st.error("❌ Error al cargar modelo")
    st.stop()

# ==========================================================
# INPUT
# ==========================================================

archivo = st.file_uploader("📁 Cargar archivo CSV del LIMS", type=["csv"])

# ==========================================================
# FUNCIONES
# ==========================================================

def extraer_valor(df, nombre):
    fila = df[df[1] == nombre]
    if fila.empty:
        return np.nan
    return fila.iloc[0, 4]

def convertir(valor):
    if isinstance(valor, str):
        try:
            return float(valor.replace(",", "."))
        except:
            return np.nan
    return valor

# ==========================================================
# BOTÓN
# ==========================================================

if archivo is not None:

    if st.button("🚀 Calcular RON"):

        with st.spinner("Procesando muestra..."):

            Recombinada = pd.read_csv(archivo, sep=";", encoding="latin1", header=None)

            try:
                celda_producto = Recombinada.loc[Recombinada[0] == "Producto", 4].values[0]
                celda_lims = Recombinada.loc[Recombinada[0] == "Número de Muestra", 4].values[0]
            except:
                st.error("❌ Formato de archivo inválido")
                st.stop()

            # VARIABLES
            datos = {
                'DENSIDAD': extraer_valor(Recombinada, "Densidad a 15ºC"),
                'IBP': extraer_valor(Recombinada, "IBP"),
                'T5': extraer_valor(Recombinada, "5% vol"),
                'T10': extraer_valor(Recombinada, "10% vol"),
                'T20': extraer_valor(Recombinada, "20% vol"),
                'T30': extraer_valor(Recombinada, "30% vol"),
                'T40': extraer_valor(Recombinada, "40% vol"),
                'T50': extraer_valor(Recombinada, "50% vol"),
                'T60': extraer_valor(Recombinada, "60% vol"),
                'T70': extraer_valor(Recombinada, "70% vol"),
                'T80': extraer_valor(Recombinada, "80% vol"),
                'T90': extraer_valor(Recombinada, "90% vol"),
                'T95': extraer_valor(Recombinada, "95% vol"),
                'PUNTO FINAL': extraer_valor(Recombinada, "Punto Final"),
                'AZUFRE': extraer_valor(Recombinada, "Azufre")
            }

            datos_convertidos = {k: convertir(v) for k, v in datos.items()}

            if celda_producto == "GASOLINA_RECOMB":

                faltantes = [
                    k for k, v in datos_convertidos.items()
                    if isinstance(v, float) and np.isnan(v)
                ]

                if faltantes:
                    st.error("❌ Datos incompletos")
                    st.warning("Faltan ensayos:")
                    st.write(", ".join(faltantes))
                    st.stop()

            df_pred = pd.DataFrame([datos])[columnas_modelo]

            for col in df_pred.columns:
                df_pred[col] = (
                    df_pred[col]
                    .astype(str)
                    .str.replace(",", ".", regex=False)
                    .astype(float)
                )

            pred_arboles = np.array([
                tree.predict(df_pred)[0]
                for tree in RF.estimators_
            ])

            ron_estimado = np.round(pred_arboles.mean(), 1)
            ron_std = pred_arboles.std()

            # ======================================================
            # SEMÁFORO
            # ======================================================

            if ron_std <= UMBRAL_METODO:
                color = "green"
                estado = "ALTA CONFIABILIDAD"
                icono = "🟢"

            elif ron_std < UMBRAL_METODO_SUP:
                color = "orange"
                estado = "CONFIABILIDAD MEDIA"
                icono = "🟡"

            else:
                color = "red"
                estado = "BAJA CONFIABILIDAD"
                icono = "🔴"

            # ======================================================
            # RESULTADO VISUAL PRO
            # ======================================================

            if celda_producto == "GASOLINA_RECOMB":

                st.markdown("---")

                col1, col2 = st.columns(2)

                
                with col1:
                    st.markdown("### 🔢 RON estimado")

                    if ron_std < UMBRAL_METODO_SUP:
                        valor = str(ron_estimado).replace(".", ",")
                        color_ron = "black"
                    else:
                        valor = "❌"
                        color_ron = "red"

                    st.markdown(
                        f"""
                        <div style="
                            text-align: center;
                            font-size: 32px;
                            font-weight: bold;
                            color: {color_ron};
                        ">
                            {valor}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                
                with col2:
                    st.markdown(f"### 📋 LIMS: {celda_lims}")

                st.markdown("---")

                # Semáforo grande
                st.markdown(
                    f"""
                    <div style="text-align:center;">
                        <h2 style="color:{color};">{icono} {estado}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:
                st.error("❌ Archivo no corresponde a GASOLINA RECOMBINADA")

