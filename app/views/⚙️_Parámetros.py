import streamlit as st

# Titulo de la página
col_title1, col_title2, col_title3 = st.columns([34, 36, 30])
with col_title2:
    st.header("⚙️ Parámetros")
    st.write("") # Espacio
    st.write("") # Espacio
    st.write("") # Espacio

# Renderiza la pestaña de hiperparámetros
def render_hyperparameters(team1_name, team2_name):
    t2col1, t2col2 = st.columns([1,1])
    with t2col1:
        player_model_conf_thresh = st.slider('Umbral de Confianza de Detección de Jugadores', min_value=0.0, max_value=1.0, value=0.6)
        detection_hyper_params = {
            0: player_model_conf_thresh,
            1: None,  # Will be set below
            2: None   # Will be set below
        }
    with t2col2:
        num_pal_colors = st.slider(label="Número de colores de paleta", min_value=1, max_value=5, step=1, value=3,
                                help="¿Cuántos colores extraer de las cajas delimitadoras de los jugadores detectados? Se utiliza para la predicción del equipo.")

    # Opciones de puntos clave del campo (debajo de las columnas principales)
    kp_col1, kp_col2 = st.columns([1,1])
    with kp_col1:
        keypoints_model_conf_thresh = st.slider('Umbral de Confianza de Detección de Puntos Clave del Campo', min_value=0.0, max_value=1.0, value=0.7)
    with kp_col2:
        keypoints_displacement_mean_tol = st.slider('Tolerancia RMSE de Desplazamiento de Puntos Clave (píxeles)', min_value=-1, max_value=100, value=7,
                                                     help="Indica la distancia promedio máxima permitida entre la posición de los puntos clave del campo en las detecciones actuales y anteriores. Se utiliza para determinar si actualizar la matriz de homografía o no.")

    # Actualizar el diccionario con los valores correctos
    detection_hyper_params[1] = keypoints_model_conf_thresh
    detection_hyper_params[2] = keypoints_displacement_mean_tol

    st.markdown("---")
    st.subheader("Opciones de Salida")
    save_processed_separately = st.checkbox(label='Guardar video procesado por separado', value=True)
    save_tactical_separately = st.checkbox(label='Guardar mapa táctico por separado', value=True)
    enable_resize = st.checkbox("Habilitar Redimensionamiento de Salida", value=False)
    if save_processed_separately or save_tactical_separately:
        output_file_name = st.text_input(label='Nombre del Archivo (Opcional)', placeholder='Ingrese el nombre del archivo de video de salida.')
    else:
        output_file_name = None
    output_width = None
    output_height = None
    if enable_resize:
        output_width = st.number_input("Ancho de Salida (px)", min_value=100, value=1280, step=100)
        output_height = st.number_input("Alto de Salida (px)", min_value=100, value=720, step=100)

    st.markdown("---")

    bcol1, bcol2 = st.columns([1,1])
    with bcol1:
        nbr_frames_no_ball_thresh = st.number_input("Umbral de reinicio del seguimiento del balón (fotogramas)", min_value=1, max_value=10000,
                                                 value=30, help="¿Después de cuántos fotogramas sin detección de balón, se debe reiniciar el seguimiento?")
        ball_track_dist_thresh = st.number_input("Umbral de distancia del seguimiento del balón (píxeles)", min_value=1, max_value=1280,
                                                    value=100, help="Distancia máxima permitida entre dos detecciones consecutivas de balón para mantener el seguimiento actual.")
        max_track_length = st.number_input("Longitud máxima del seguimiento del balón (Núm. detecciones)", min_value=1, max_value=1000,
                                                    value=35, help="Número máximo total de detecciones de balón para mantener en el historial de seguimiento")
        ball_track_hyperparams = {
            0: nbr_frames_no_ball_thresh,
            1: ball_track_dist_thresh,
            2: max_track_length
        }
    with bcol2:
        st.write("Opciones de anotación:")
        bcol21t, bcol22t = st.columns([1,1])
        with bcol21t:
            show_k = st.toggle(label="Mostrar Detecciones de Puntos Clave", value=False)
            show_p = st.toggle(label="Mostrar Detecciones de Jugadores", value=True)
        with bcol22t:
            show_pal = st.toggle(label="Mostrar Paletas de Color", value=True)
            show_b = st.toggle(label="Mostrar Seguimientos del Balón", value=True)
        plot_hyperparams = {
            0: show_k,
            1: show_pal,
            2: show_b,
            3: show_p
        }
        st.markdown('---')
        bcol21, bcol22, bcol23, bcol24 = st.columns([1.5,1,1,1])
        with bcol21:
            st.write('')
        with bcol22:
            ready = True if (team1_name == '') or (team2_name == '') else False
            start_detection = st.button(label='Iniciar Detección', disabled=ready)
        with bcol23:
            stop_detection = st.button(label='Detener Detección')
        with bcol24:
            st.write('')

    return (detection_hyper_params, num_pal_colors, save_processed_separately, save_tactical_separately,
            output_file_name, enable_resize, output_width, output_height, ball_track_hyperparams, plot_hyperparams,
            start_detection, stop_detection)

# Ejecutar la configuración de parámetros y detección
if "input_vide_file" not in st.session_state:
    st.error("Primero carga un video en la pestaña 'Carga de Video'.")
elif "colors_dic" not in st.session_state:
    st.error("Primero configura los colores de equipo en la página de Configuración de Colores.")
else:
    team1_name = st.session_state.team1_name
    team2_name = st.session_state.team2_name
    (detection_hyper_params, num_pal_colors, save_processed_separately, save_tactical_separately,
     output_file_name, enable_resize, output_width, output_height, ball_track_hyperparams, plot_hyperparams,
     start_detection, stop_detection) = render_hyperparameters(team1_name, team2_name)

    import cv2
    tempf = st.session_state.tempf
    cap = cv2.VideoCapture(tempf.name)

    if start_detection and not stop_detection:
        from detection import detect
        st.toast(f'¡Detección Iniciada!')
        save_combined = False  # No longer an option, always False
        model_players = st.session_state.model_players
        model_keypoints = st.session_state.model_keypoints
        colors_dic = st.session_state.colors_dic
        color_list_lab = st.session_state.color_list_lab
        stframe = st.empty()
        detect(cap, stframe, output_file_name, save_processed_separately, save_tactical_separately, save_combined, model_players, model_keypoints,
               detection_hyper_params, ball_track_hyperparams, plot_hyperparams,
               num_pal_colors, colors_dic, color_list_lab, enable_resize, output_width, output_height)
    else:
        try:
            cap.release()
        except:
            pass
