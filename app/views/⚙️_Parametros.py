import cv2
import streamlit as st

# Titulo de la página
col_title1, col_title2, col_title3 = st.columns([40, 30, 30])
with col_title2:
    st.header("⚙️ Parámetros")
    st.write("") # Espacio

# Renderiza la pestaña de hiperparámetros
def render_hyperparameters():
    # Expander para agrupar las configuraciones de detección
    with st.expander("Configuraciones de Detección", expanded=True):
        # Umbral de confianza para detección de jugadores
        player_model_conf_thresh = st.slider(
            'Umbral de Confianza de Detección de Jugadores',
            min_value=0.0,
            max_value=1.0,
            value=0.4
            )
        
        # Inicializar diccionario de parámetros de detección
        detection_hyper_params = {
            0: player_model_conf_thresh,
            1: None,  # Will be set below
            2: None   # Will be set below
        }

        # Umbral de confianza para detección de puntos clave del campo
        keypoints_model_conf_thresh = st.slider(
            'Umbral de Confianza de Detección de Puntos Clave del Campo',
            min_value=0.0,
            max_value=1.0,
            value=0.7
            )

        # Tolerancia de desplazamiento para puntos clave (usada para actualizar homografía)
        keypoints_displacement_mean_tol = st.slider(
            'Tolerancia RMSE de Desplazamiento de Puntos Clave (píxeles)',
            min_value=-1,
            max_value=100,
            value=7,
            help="Indica la distancia promedio máxima permitida entre la posición de los puntos clave del campo en las detecciones actuales y anteriores."
            )

        # Actualizar el diccionario con los valores correctos
        detection_hyper_params[1] = keypoints_model_conf_thresh
        detection_hyper_params[2] = keypoints_displacement_mean_tol

    # Expander para opciones de visualización
    with st.expander("Opciones de Visualización", expanded=True):
        vis_col1, vis_col2 = st.columns([1,1])
        
        # Opciones de anotación
        with vis_col1:
            st.subheader("Opciones de Anotación")
            show_p = st.toggle(label="Mostrar Detecciones de Jugadores", value=True)
            show_k = st.toggle(label="Mostrar Detecciones de Puntos Clave", value=False)
            show_b = st.toggle(label="Mostrar Seguimientos del Balón", value=False)
            plot_hyperparams = {
                0: show_k,
                1: show_b,
                2: show_p
            }
        
        # Opciones de salida de video
        with vis_col2:
            st.subheader("Opciones de Salida")
            save_processed_separately = st.checkbox(label='Guardar juego procesado', value=True)
            save_tactical_separately = st.checkbox(label='Guardar mapa táctico', value=True)
            if save_processed_separately or save_tactical_separately:
                output_file_name = st.text_input(label='Nombre del Archivo (Opcional)', placeholder='Ingrese el nombre del archivo de salida.')
            else:
                output_file_name = None

    # Opciones del balón
    nbr_frames_no_ball_thresh = 30
    ball_track_dist_thresh = 100
    max_track_length = 35
    ball_track_hyperparams = {
        0: nbr_frames_no_ball_thresh,
        1: ball_track_dist_thresh,
        2: max_track_length
    }

    # CSS para quitar redondeo de contenedores e imágenes
    st.markdown("""<style> .stContainer { border-radius: 0 !important; } img { border-radius: 0 !important; } </style>""", unsafe_allow_html=True)

    # Contenedor para los botones de detección (full width con borde)
    with st.container(border=True):
        btn_col1, btn_col2, btn_col3 = st.columns([25, 37, 38])
        with btn_col2:
            start_detection = st.button(label='Iniciar Detección')
        with btn_col3:
            stop_detection = st.button(label='Detener Detección')

    return (
        detection_hyper_params,
        save_processed_separately,
        save_tactical_separately,
        output_file_name,
        ball_track_hyperparams,
        plot_hyperparams,
        start_detection,
        stop_detection
        )

# Ejecutar la configuración de parámetros y detección
if "input_vide_file" not in st.session_state:
    st.error("Primero carga un video en la pestaña 'Carga de Video'.")
else:
    (
        detection_hyper_params,
        save_processed_separately,
        save_tactical_separately,
        output_file_name,
        ball_track_hyperparams,
        plot_hyperparams,
        start_detection,
        stop_detection
    ) = render_hyperparameters()

    tempf = st.session_state.tempf
    cap = cv2.VideoCapture(tempf.name)

    if start_detection and not stop_detection:
        from detection import detect
        st.toast(f'¡Detección Iniciada!')
        save_combined = False # No longer an option, always False
        model_players = st.session_state.model_players
        model_keypoints = st.session_state.model_keypoints
        colors_dic = {} # Will be set in detection
        
        # Crear placeholders para los videos separados con espacio
        col_space, col_game, col_tactical = st.columns([15, 50, 35])
        
        # Procesado del juego
        with col_game:
            st.subheader("Video del Juego")
            stframe_game = st.empty()
        
        # Procesado del mapa táctico
        with col_tactical:
            st.subheader("Mapa Táctico")
            stframe_tactical = st.empty()
        
        detect(
            cap,
            stframe_game,
            stframe_tactical,
            output_file_name,
            save_processed_separately,
            save_tactical_separately,
            save_combined,
            model_players,
            model_keypoints,
            detection_hyper_params,
            ball_track_hyperparams,
            plot_hyperparams,
            colors_dic
            )
    else:
        try:
            cap.release()
        except:
            pass
