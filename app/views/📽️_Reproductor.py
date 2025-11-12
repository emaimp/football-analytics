import os
import tempfile
import streamlit as st

# Titulo de la página
col_title1, col_title2, col_title3 = st.columns([34, 36, 30])
with col_title2:
    st.header("📽️ Reproductor")
    st.write("") # Espacio
    st.write("") # Espacio
    st.write("") # Espacio

# Cargar y reproducir videos locales
def render_local_video():
    uploaded_videos = st.file_uploader(
        "Selecciona uno o más videos para reproducir.",
        type=['mp4', 'avi', 'mov', 'm4v', 'mp3', 'wav'],
        accept_multiple_files=True,
        help="Selecciona los videos a reproducir."
    )
    st.divider()

    if uploaded_videos:
        # Mostrar videos en columnas (máximo 2 por fila)
        cols = st.columns(2)
        for i, video_file in enumerate(uploaded_videos):
            col_idx = i % 2
            with cols[col_idx]:
                st.markdown(f'<p style="text-align: center;">{video_file.name}</p>', unsafe_allow_html=True)
                st.markdown(f'<p style="text-align: center;">Tamaño: {video_file.size/1024/1024:.1f} MB</p>', unsafe_allow_html=True)
                # Guardar archivo temporalmente
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.name)[1]) as temp_file:
                    temp_file.write(video_file.getvalue())
                    temp_file_path = temp_file.name

                # Reproducir video
                st.video(temp_file_path)

# Ejecutar la página
render_local_video()
