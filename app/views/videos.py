import os
import re
import streamlit as st

def render_local_video_player_tab():
    st.header("Reproductor de Videos")

    # Sección para subir videos
    st.subheader("📤 Videos para reproducir")
    uploaded_videos = st.file_uploader(
        "Selecciona uno o más videos para reproducir",
        type=['mp4', 'avi', 'mov', 'm4v', 'mp3', 'wav'],
        accept_multiple_files=True,
        help="Puedes seleccionar múltiples videos de tu dispositivo para reproducirlos aquí."
    )

    if uploaded_videos:
        st.subheader("🎬 Videos Subidos")

        # Mostrar videos en columnas (máximo 2 por fila)
        cols = st.columns(2)
        for i, video_file in enumerate(uploaded_videos):
            col_idx = i % 2
            with cols[col_idx]:
                st.markdown(f"**🎥 {video_file.name}**")
                st.caption(f"Tamaño: {video_file.size/1024/1024:.1f} MB")
                st.video(video_file)
                # Opción para descargar el video subido
                st.download_button(
                    label="📥 Descargar",
                    data=video_file.getvalue(),
                    file_name=video_file.name,
                    mime=video_file.type,
                    key=f"download_{i}"
                )
                # Espacio entre videos
                if i < len(uploaded_videos) - 1:
                    st.markdown("---")

    st.markdown("---")

    # Sección de videos procesados (existente)
    st.header("Últimos Videos Procesados")

    if os.path.exists('outputs/'):
        video_files = [f for f in os.listdir('outputs/') if f.endswith('.mp4')]
        processed_files = [f for f in video_files if '_processed.mp4' in f]
        tactical_files = [f for f in video_files if '_tactical.mp4' in f]

        if processed_files and tactical_files:
            # Extraer números de los archivos
            processed_nums = []
            for f in processed_files:
                match = re.search(r'detect_(\d+)_processed\.mp4', f)
                if match:
                    processed_nums.append(int(match.group(1)))

            tactical_nums = []
            for f in tactical_files:
                match = re.search(r'detect_(\d+)_tactical\.mp4', f)
                if match:
                    tactical_nums.append(int(match.group(1)))

            if processed_nums and tactical_nums:
                # Encontrar el máximo N común
                common_nums = set(processed_nums) & set(tactical_nums)
                if common_nums:
                    max_n = max(common_nums)
                    processed_path = f'outputs/detect_{max_n}_processed.mp4'
                    tactical_path = f'outputs/detect_{max_n}_tactical.mp4'

                    # Mostrar en columnas
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.subheader("Video del Juego Procesado")
                        st.video(processed_path)
                    with col2:
                        st.subheader("Mapa Táctico")
                        st.video(tactical_path)
                else:
                    st.write("No se encontraron videos procesados y tácticos con números coincidentes.")
            else:
                st.write("No se pudieron extraer números de los archivos de video.")
        else:
            st.write("No se encontraron videos procesados (_processed.mp4) o tácticos (_tactical.mp4).")
    else:
        st.write("La carpeta 'outputs' no existe. Procesa un video primero para ver los resultados.")
