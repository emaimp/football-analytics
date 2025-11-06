import os
import cv2
import time
import numpy as np
import streamlit as st
from core import annotations, config, homography, output, prediction

"""
Función principal para detectar y procesar frames de video de fútbol.
Realiza detección de objetos, transformación de coordenadas, predicción de equipos y anotaciones.
"""
def detect(cap, stframe, output_file_name, save_processed_separately, save_tactical_separately, save_combined, model_players, model_keypoints,
            hyper_params, ball_track_hyperparams, plot_hyperparams, num_pal_colors, colors_dic, color_list_lab,
            enable_resize, output_width, output_height):

    # Extraer parámetros de visualización
    show_k = plot_hyperparams[0] # Mostrar keypoints
    show_pal = plot_hyperparams[1] # Mostrar paletas
    show_b = plot_hyperparams[2] # Mostrar balón
    show_p = plot_hyperparams[3] # Mostrar jugadores

    # Extraer parámetros de hiperparámetros
    p_conf = hyper_params[0] # Confianza para detección de jugadores
    k_conf = hyper_params[1] # Confianza para detección de keypoints
    k_d_tol = hyper_params[2] # Tolerancia de distancia para keypoints

    # Extraer parámetros de seguimiento del balón
    nbr_frames_no_ball_thresh = ball_track_hyperparams[0] # Umbral de frames sin balón
    ball_track_dist_thresh = ball_track_hyperparams[1] # Umbral de distancia para seguimiento
    max_track_length = ball_track_hyperparams[2] # Longitud máxima de seguimiento

    # Número de colores por equipo
    nbr_team_colors = len(list(colors_dic.values())[0])

    # Generar nombre de archivo si es necesario
    if (save_processed_separately or save_tactical_separately or save_combined) and (output_file_name is None or len(str(output_file_name)) == 0):
        output_file_name = config.generate_file_name()

    # Asegurar que el directorio de salidas existe
    os.makedirs('./outputs/', exist_ok=True)

    # Leer imagen del mapa táctico
    tac_map = cv2.imread('app\\assets\\campo.png')
    map_height, map_width, _ = tac_map.shape # Obtener correctamente las dimensiones del mapa

    # Crear escritores de video de salida
    processed_output = None # Inicializar escritor de video procesado a None
    tactical_output = None # Inicializar escritor de video táctico a None
    combined_output = None # Inicializar escritor de video combinado a None

    # Crear barra de progreso
    tot_nbr_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) # Obtener FPS original del video
    st_prog_bar = st.progress(0, text='Detección iniciando.')

    # Obtener diccionarios de etiquetas y posiciones
    keypoints_map_pos, classes_names_dic, labels_dic = config.get_labels_dics()

    # Variable para registrar el tiempo cuando procesamos el último frame
    prev_frame_time = 0
    # Variable para registrar el tiempo en el que procesamos el frame actual
    new_frame_time = 0

    # Almacenar el historial de seguimiento del balón
    ball_track_history = {'src': [], # Posiciones fuente del balón
                          'dst': [] # Posiciones destino del balón
    }
    nbr_frames_no_ball = 0 # Contador de frames sin balón

    # Inicializar variables de estado de homografía
    detected_labels_prev = None
    detected_labels_src_pts_prev = None

    # Bucle sobre los frames del video de entrada
    for frame_nbr in range(1, tot_nbr_frames + 1):

        # Actualizar barra de progreso
        percent_complete = int(frame_nbr / (tot_nbr_frames) * 100)
        st_prog_bar.progress(percent_complete, text=f"Detección en progreso ({percent_complete}%)")

        # Leer un frame del video
        success, frame = cap.read()

        # Reiniciar imagen del mapa táctico para cada nuevo frame
        tac_map_copy = tac_map.copy()

        # Reiniciar historial de balón si no se detecta por muchos frames
        if nbr_frames_no_ball > nbr_frames_no_ball_thresh:
            ball_track_history['dst'] = []
            ball_track_history['src'] = []

        if success:
            ### Detección de Objetos & Transformación de Coordenadas ###

            # Ejecutar inferencia de YOLOv8 para jugadores en el frame
            results_players = model_players(frame, conf=p_conf)
            # Ejecutar inferencia de YOLOv8 para keypoints del campo en el frame
            results_keypoints = model_keypoints(frame, conf=k_conf)

            # Extraer información de detecciones
            bboxes_p = results_players[0].boxes.xyxy.cpu().numpy() # Bounding boxes de jugadores, árbitros y balón detectados (x,y,x,y)
            bboxes_p_c = results_players[0].boxes.xywh.cpu().numpy() # Bounding boxes de jugadores, árbitros y balón detectados (x,y,w,h)
            labels_p = list(results_players[0].boxes.cls.cpu().numpy()) # Lista de etiquetas de jugadores, árbitros y balón detectados
            confs_p = list(results_players[0].boxes.conf.cpu().numpy()) # Nivel de confianza de jugadores, árbitros y balón detectados

            bboxes_k = results_keypoints[0].boxes.xyxy.cpu().numpy() # Bounding boxes de keypoints del campo detectados (x,y,x,y)
            bboxes_k_c = results_keypoints[0].boxes.xywh.cpu().numpy() # Bounding boxes de keypoints del campo detectados (x,y,w,h)
            labels_k = list(results_keypoints[0].boxes.cls.cpu().numpy()) # Lista de etiquetas de keypoints del campo detectados

            # Convertir etiquetas numéricas detectadas a etiquetas alfabéticas
            detected_labels = [classes_names_dic[i] for i in labels_k]

            # Extraer coordenadas de keypoints del campo detectados en el frame actual
            detected_labels_src_pts = np.array([list(np.round(bboxes_k_c[i][:2]).astype(int)) for i in range(bboxes_k_c.shape[0])])

            # Obtener las coordenadas de keypoints del campo detectados en el mapa táctico
            detected_labels_dst_pts = np.array([keypoints_map_pos[i] for i in detected_labels])

            # Calcular matriz de transformación de homografía
            homog, update_homography, detected_labels_prev, detected_labels_src_pts_prev = homography.calculate_homography(
                detected_labels, detected_labels_src_pts, detected_labels_dst_pts,
                detected_labels_prev, detected_labels_src_pts_prev, k_d_tol, frame_nbr
            )

            # Inicializar posiciones predichas
            pred_dst_pts = None # Puntos destino de jugadores
            detected_ball_src_pos = None # Posición fuente del balón detectado
            detected_ball_dst_pos = None # Posición destino del balón detectado

            if homog is not None:
                # Obtener información de bounding boxes (x,y,w,h) de jugadores detectados (etiqueta 0)
                bboxes_p_c_0 = bboxes_p_c[[i == 0 for i in labels_p], :]
                # Obtener información de bounding boxes (x,y,w,h) del balón detectado (etiqueta 2)
                bboxes_p_c_2 = bboxes_p_c[[i == 2 for i in labels_p], :]

                # Obtener coordenadas de jugadores detectados en el frame (x_centro, y_centro+h/2)
                detected_ppos_src_pts = bboxes_p_c_0[:, :2] + np.array([[0] * bboxes_p_c_0.shape[0], bboxes_p_c_0[:, 3] / 2]).transpose()
                # Obtener coordenadas del primer balón detectado (x_centro, y_centro)
                detected_ball_src_pos = bboxes_p_c_2[0, :2] if bboxes_p_c_2.shape[0] > 0 else None

                # Actualizar contador de frames sin balón
                if detected_ball_src_pos is None:
                    nbr_frames_no_ball += 1
                else:
                    nbr_frames_no_ball = 0

                # Transformar coordenadas de jugadores
                pred_dst_pts = homography.transform_points(homog, detected_ppos_src_pts)

                # Transformar coordenadas del balón
                if detected_ball_src_pos is not None:
                    detected_ball_dst_pos = homography.transform_points(homog, [detected_ball_src_pos])[0]

                    # Actualizar seguimiento del balón
                    if show_b:
                        ball_track_history = homography.update_ball_tracking(ball_track_history, detected_ball_src_pos, detected_ball_dst_pos,
                                                                  ball_track_dist_thresh, max_track_length)

            ### Predicción de Equipos de Jugadores ###

            # Extraer paletas de colores de jugadores
            obj_palette_list = prediction.extract_player_palettes(frame, bboxes_p, labels_p, num_pal_colors)
            # Calcular características de distancia para predicción de equipos
            players_distance_features = prediction.calculate_distance_features(obj_palette_list, color_list_lab)
            # Predecir equipos de jugadores
            players_teams_list = prediction.predict_teams(players_distance_features, nbr_team_colors)

            ### Frame Actualizado & Mapa Táctico Con Anotaciones ###

            # Anotar el frame con detecciones
            annotated_frame = annotations.annotate_frame(frame, bboxes_p, labels_p, confs_p, players_teams_list, colors_dic,
                                           obj_palette_list, labels_dic, show_pal, show_p, show_k, bboxes_k)

            # Anotar el mapa táctico con posiciones de jugadores y balón
            tac_map_copy = annotations.annotate_tactical_map(tac_map_copy, pred_dst_pts, detected_ball_dst_pos, players_teams_list, colors_dic)
            # Dibujar trayectoria del balón en el mapa táctico
            tac_map_copy = annotations.draw_ball_trajectory(tac_map_copy, ball_track_history)

            # Guardar videos separados si está habilitado
            processed_output = output.write_processed_video(processed_output, annotated_frame, output_file_name, fps, save_processed_separately)
            tactical_output = output.write_tactical_video(tactical_output, tac_map_copy, output_file_name, fps, save_tactical_separately)

            # Calcular FPS actual
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time

            # Combinar frame anotado y mapa táctico
            final_img = annotations.combine_frames(annotated_frame, tac_map_copy, enable_resize, output_width, output_height)
            # Agregar texto de FPS a la imagen final
            final_img = annotations.add_fps_text(final_img, fps)

            # Guardar video combinado si está habilitado
            combined_output = output.write_combined_video(combined_output, final_img, output_file_name, fps, save_combined)

            # Mostrar el frame anotado en Streamlit
            stframe.image(final_img, channels="BGR")
            # cv2.imshow("YOLOv8 Inference", frame) # Opción alternativa para mostrar con OpenCV

    # Liberar escritores de video de salida y devolver nombres de archivos
    st_prog_bar.empty()
    processed_name, tactical_name, combined_name = output.release_video_writers(
        processed_output, tactical_output, combined_output,
        save_processed_separately, save_tactical_separately, save_combined, output_file_name
    )
    return True, processed_name, tactical_name, combined_name
