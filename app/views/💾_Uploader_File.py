import tempfile
import streamlit as st

# Titulo de la página
col_title1, col_title2, col_title3 = st.columns([34, 36, 30])
with col_title2:
    st.header("💾 Carga de Video")
    st.write("") # Espacio
    st.write("") # Espacio
    st.write("") # Espacio

# Cargar video y nombres de equipos
def video_uploader():
    input_vide_file = st.file_uploader('Selecciona un video para procesar.', type=['mp4','mov', 'avi', 'm4v', 'asf'])

    # Si se sube un nuevo video, procesarlo
    if input_vide_file:
        tempf = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        tempf.write(input_vide_file.read())
        video_display = open(tempf.name, 'rb')
        video_bytes = video_display.read()

        st.subheader('Video de Entrada')
        st.video(video_bytes)

        st.divider() # Divisor

        # Nombres de equipos
        st.subheader("Nombres de Equipos")
        col1, col2 = st.columns(2)
        with col1:
            team1_name = st.text_input(label='Primer Equipo', value='')
        with col2:
            team2_name = st.text_input(label='Segundo Equipo', value='')

        st.divider() # Divisor

        return input_vide_file, tempf, video_bytes, team1_name, team2_name

    # Si ya hay un video cargado, mostrarlo
    elif 'video_bytes' in st.session_state:
        st.subheader('Video de Entrada (Cargado)')
        st.video(st.session_state.video_bytes)

        st.divider() # Divisor

        # Nombres de equipos (editables)
        st.subheader("Nombres de Equipos")
        col1, col2 = st.columns(2)
        with col1:
            team1_name = st.text_input(label='Primer Equipo', value=st.session_state.get('team1_name', ''))
        with col2:
            team2_name = st.text_input(label='Segundo Equipo', value=st.session_state.get('team2_name', ''))

        st.divider() # Divisor

        return st.session_state.input_vide_file, st.session_state.tempf, st.session_state.video_bytes, team1_name, team2_name

    # No hay video
    else:
        return None, None, None, None, None

# Ejecutar la función de carga
input_vide_file, tempf, video_bytes, team1_name, team2_name = video_uploader()

if input_vide_file:
    st.session_state.input_vide_file = input_vide_file
    st.session_state.tempf = tempf
    st.session_state.video_bytes = video_bytes
    st.session_state.team1_name = team1_name
    st.session_state.team2_name = team2_name
    st.success("Video cargado correctamente. Ahora puedes configurar los colores y parámetros en sus respectivas pestañas.")
else:
    st.info("Por favor, carga un video para continuar.")
