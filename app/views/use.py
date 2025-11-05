import streamlit as st

def render_how_to_use_tab():
    st.header(':blue[Bienvenido!]')
    st.subheader('Funcionalidades de la Aplicación:', divider='blue')
    st.markdown("""
                1. Detección de jugadores, árbitro y balón.
                2. Predicción del equipo.
                3. Estimación de las posiciones de jugadores y balón en el campo.
                """)
    st.subheader('Cómo usar:', divider='blue')
    st.markdown("""
                1. Sube un video para analizar en el menú de la barra lateral.
                2. Ingresa los nombres de los equipos en el menú de la barra lateral.
                3. Accede a la pestaña "Colores de Equipo".
                4. Selecciona un fotograma donde se puedan detectar jugadores y porteros.
                5. Sigue las instrucciones en la página para seleccionar los colores de cada equipo.
                6. Accede a la pestaña "Parámetros del Modelo", ajusta las configuraciones (se recomiendan las predeterminadas).
                7. Ejecuta la Detección.
                8. Si se seleccionó la opción "guardar salidas", el video guardado se puede encontrar en el directorio "outputs"
                """)
    st.write("Versión 0.0.2")
