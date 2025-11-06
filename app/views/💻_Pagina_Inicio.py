import streamlit as st

def render_how_to_use():
    st.header(':blue[Bienvenido ⚽]')
    st.write("") # Espacio
    st.subheader('Funcionalidades de la Aplicación:')
    st.markdown("""
                1. Detección de jugadores, árbitro y balón.
                2. Predicción del equipo.
                3. Estimación de las posiciones de jugadores y balón en el campo.
                """)
    st.write("") # Espacio
    st.subheader('Cómo usar:')
    st.markdown("""
                1. Ve a la página "Carga de Video" y sube un video para analizar.
                2. Ingresa los nombres de los equipos en la carga de video.
                3. Accede a la página "Configuración de Colores".
                4. Selecciona un fotograma donde se puedan detectar jugadores y porteros.
                5. Sigue las instrucciones en la página para seleccionar los colores de cada equipo.
                6. Accede a la página "Configuración de Parámetros", ajusta las configuraciones (se recomiendan las predeterminadas).
                7. Ejecuta la Detección.
                8. Si se seleccionó la opción "guardar salidas", el video guardado se puede encontrar en el directorio "outputs"
                """)
    st.write("") # Espacio
    st.caption("(Solo funciona con videos con vista táctica)")

# Ejecutar la página
render_how_to_use()
