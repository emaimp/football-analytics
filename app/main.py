import cv2
import tempfile
import streamlit as st
from ultralytics import YOLO
from detection import detect
from views.use import render_how_to_use_tab
from views.colors import render_team_colors_tab
from views.videos import render_local_video_player_tab
from views.parameters import render_hyperparameters_tab

def main():
    st.set_page_config(page_title="Aplicación Web Potenciada por IA para Análisis Táctico de Fútbol", layout="wide", initial_sidebar_state="expanded")
    st.title("Computer Vision - Football")
    st.subheader(":red[Solo funciona con videos en vista táctica]")

    # Configuración de la Barra Lateral
    st.sidebar.subheader("Video para el análisis")
    input_vide_file = st.sidebar.file_uploader('Carga de video.', type=['mp4','mov', 'avi', 'm4v', 'asf'])

    # Configuración de la Página - Las pestañas siempre se muestran
    tab1, tab2, tab3, tab4 = st.tabs(["Inicio", "Colores de Equipo", "Parámetros del Modelo", "Reproductor de Video"])

    with tab1:
        render_how_to_use_tab()

    if not input_vide_file:
        st.sidebar.text('Por favor, seleccione un archivo.')
        with tab2:
            st.info("💡 Seleccione un video para configurar los colores de equipo.")
        with tab3:
            st.info("⚙️ Seleccione un video para ajustar los parámetros de detección.")
        with tab4:
            st.info("🎥 Seleccione un video para ver los resultados procesados.")
        return

    tempf = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    tempf.write(input_vide_file.read())
    video_display = open(tempf.name, 'rb')
    video_bytes = video_display.read()

    st.sidebar.text('Video de entrada')
    st.sidebar.video(video_bytes)

    # Cargar el modelo de detección de jugadores YOLOv8
    model_players = YOLO("../models/Yolo8L Players/weights/best.pt")
    # Cargar el modelo de detección de puntos clave del campo YOLOv8
    model_keypoints = YOLO("../models/Yolo8M Field Keypoints/weights/best.pt")

    # Información predeterminada del equipo para videos subidos
    selected_team_info = {
        "team1_name": "",
        "team2_name": "",
        "team1_p_color": '#FFFFFF',
        "team1_gk_color": '#000000',
        "team2_p_color": '#FFFFFF',
        "team2_gk_color": '#000000',
    }

    st.sidebar.markdown('---')
    st.sidebar.subheader("Nombres de Equipos")
    team1_name = st.sidebar.text_input(label='Nombre del Primer Equipo', value=selected_team_info["team1_name"])
    team2_name = st.sidebar.text_input(label='Nombre del Segundo Equipo', value=selected_team_info["team2_name"])
    st.sidebar.markdown('---')

    with tab2:
        colors_dic, color_list_lab = render_team_colors_tab(tempf, model_players, team1_name, team2_name, selected_team_info)

    with tab3:
        (detection_hyper_params, num_pal_colors, save_processed_separately, save_tactical_separately,
         output_file_name, enable_resize, output_width, output_height, ball_track_hyperparams, plot_hyperparams,
         start_detection, stop_detection) = render_hyperparameters_tab(team1_name, team2_name)

    stframe = st.empty()
    cap = cv2.VideoCapture(tempf.name)

    if start_detection and not stop_detection:
        st.toast(f'¡Detección Iniciada!')
        save_combined = False  # No longer an option, always False
        detect(cap, stframe, output_file_name, save_processed_separately, save_tactical_separately, save_combined, model_players, model_keypoints,
               detection_hyper_params, ball_track_hyperparams, plot_hyperparams,
               num_pal_colors, colors_dic, color_list_lab, enable_resize, output_width, output_height)
    else:
        try:
            # Release the video capture object and close the display window
            cap.release()
        except:
            pass

    with tab4:
        render_local_video_player_tab()

if __name__=='__main__':
    try:
        main()
    except SystemExit:
        pass
