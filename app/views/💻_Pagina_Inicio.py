import streamlit as st

# Configuración de inicio
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
    )

# Banner
col_banner1, col_banner2, col_banner3 = st.columns([37, 33, 30])
with col_banner2:
    st.write("") # Espacio
    st.write("") # Espacio
    st.image("app/assets/logo.png", width=360)

# Título de la empresa
col_title1, col_title2, col_title3 = st.columns([34, 36, 30])
with col_title2:
    st.header("Electrodomésticos S.A")
    st.write("") # Espacio
    st.write("") # Espacio
    st.write("") # Espacio

# Columna del texto
col_inf1, col_inf2, col_inf3 = st.columns([20, 60, 20])
with col_inf2:
    st.markdown(
        """
        <style>
        .text-box {
            border: 2px solid #ffffff !important;
            padding: 10px;
            background-color: rgba(0,0,0,0.1);
            text-align: center;
        }
        </style>
        <div class="text-box">
        Esta aplicación permite identificar patrones de ventas,
        medir las ganancias y desempeño de los productos,
        asi como hacer proyecciones de rendimiento futuro 📉📈.
        </div>
        """,
        unsafe_allow_html=True
    )
