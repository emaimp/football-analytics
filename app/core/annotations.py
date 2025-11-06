import cv2
import numpy as np

def annotate_frame(frame, bboxes_p, labels_p, confs_p, players_teams_list, colors_dic,
                  obj_palette_list, labels_dic, show_pal, show_p, show_k, bboxes_k):
    """
    Anota el frame con bounding boxes, colores de equipo y etiquetas de texto.
    Args:
        frame: Frame de entrada
        bboxes_p: Bounding boxes de jugadores
        labels_p: Etiquetas de detección
        confs_p: Puntajes de confianza
        players_teams_list: Índices de equipos predichos
        colors_dic: Diccionario de colores de equipos
        obj_palette_list: Paletas de colores de jugadores
        labels_dic: Diccionario de nombres de etiquetas
        show_pal: Si mostrar paletas de colores
        show_p: Si mostrar anotaciones de jugadores
        show_k: Si mostrar bounding boxes de keypoints
        bboxes_k: Bounding boxes de keypoints
    Returns:
        annotated_frame: Frame con anotaciones
    """
    annotated_frame = frame.copy()
    palette_box_size = 10 # Establecer tamaño de caja de color en píxeles (para visualización)
    j = 0 # Inicializando contador de jugadores detectados

    # Bucle sobre todos los objetos detectados
    for i in range(bboxes_p.shape[0]):
        conf = confs_p[i] # Obtener confianza del objeto detectado actual
        if labels_p[i] == 0: # Mostrar anotación para jugadores detectados (etiqueta 0)
            # Mostrar paleta de colores extraída para cada jugador detectado
            if show_pal and j < len(obj_palette_list):
                palette = obj_palette_list[j] # Obtener paleta de colores del jugador detectado
                for k, c in enumerate(palette):
                    c_bgr = c[::-1]  # Convertir color a BGR
                    annotated_frame = cv2.rectangle(annotated_frame, (int(bboxes_p[i, 2]) + 3, # Agregar anotación de paleta de colores en el frame
                                                            int(bboxes_p[i, 1]) + k * palette_box_size),
                                                            (int(bboxes_p[i, 2]) + palette_box_size,
                                                            int(bboxes_p[i, 1]) + (palette_box_size) * (k + 1)),
                                                            c_bgr, -1)

            team_name = list(colors_dic.keys())[players_teams_list[j]] # Obtener predicción de equipo del jugador detectado
            color_rgb = colors_dic[team_name][0] # Obtener color de equipo del jugador detectado
            color_bgr = color_rgb[::-1] # Convertir color a bgr
            if show_p:
                annotated_frame = cv2.rectangle(annotated_frame, (int(bboxes_p[i, 0]), int(bboxes_p[i, 1])), # Agregar anotaciones de bbox con colores de equipo
                                                (int(bboxes_p[i, 2]), int(bboxes_p[i, 3])), color_bgr, 1)

                annotated_frame = cv2.putText(annotated_frame, team_name + f" {conf:.2f}", # Agregar anotaciones de nombre de equipo
                            (int(bboxes_p[i, 0]), int(bboxes_p[i, 1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            color_bgr, 2)

            j += 1  # Actualizar contador de jugadores
        else:  # Mostrar anotación para otras detecciones (etiqueta 1, 2)
            annotated_frame = cv2.rectangle(annotated_frame, (int(bboxes_p[i, 0]), int(bboxes_p[i, 1])), # Agregar anotaciones de bbox de color blanco
                                            (int(bboxes_p[i, 2]), int(bboxes_p[i, 3])), (255, 255, 255), 1)
            annotated_frame = cv2.putText(annotated_frame, labels_dic[labels_p[i]] + f" {conf:.2f}", # Agregar anotaciones de texto de etiqueta de color blanco
                        (int(bboxes_p[i, 0]), int(bboxes_p[i, 1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2)

    # Anotar keypoints si está habilitado
    if show_k:
        for i in range(bboxes_k.shape[0]):
            annotated_frame = cv2.rectangle(annotated_frame, (int(bboxes_k[i, 0]), int(bboxes_k[i, 1])), # Agregar anotaciones de bbox con colores de equipo
                                        (int(bboxes_k[i, 2]), int(bboxes_k[i, 3])), (0, 0, 0), 1)

    return annotated_frame

def annotate_tactical_map(tac_map_copy, pred_dst_pts, detected_ball_dst_pos, players_teams_list, colors_dic):
    """
    Anota el mapa táctico con posiciones de jugadores y balón.
    Args:
        tac_map_copy: Copia del mapa táctico
        pred_dst_pts: Posiciones de jugadores en el mapa táctico
        detected_ball_dst_pos: Posición del balón en el mapa táctico
        players_teams_list: Índices de equipos predichos
        colors_dic: Diccionario de colores de equipos
    Returns:
        annotated_tactical_map: Mapa táctico con anotaciones
    """
    annotated_tactical_map = tac_map_copy.copy()
    ball_color_bgr = (0, 0, 255) # Color (BGR) para anotación del balón en el mapa táctico

    # Anotar posiciones de jugadores
    if pred_dst_pts is not None:
        for j, pt in enumerate(pred_dst_pts):
            team_name = list(colors_dic.keys())[players_teams_list[j]]
            color_rgb = colors_dic[team_name][0]
            color_bgr = color_rgb[::-1]
            annotated_tactical_map = cv2.circle(annotated_tactical_map, (int(pt[0]), int(pt[1])),
                                        radius=5, color=color_bgr, thickness=-1)
            annotated_tactical_map = cv2.circle(annotated_tactical_map, (int(pt[0]), int(pt[1])),
                                        radius=5, color=(0, 0, 0), thickness=1)

    # Anotar posición del balón
    if detected_ball_dst_pos is not None:
        annotated_tactical_map = cv2.circle(annotated_tactical_map, (int(detected_ball_dst_pos[0]),
                                            int(detected_ball_dst_pos[1])), radius=5,
                                            color=ball_color_bgr, thickness=3)

    return annotated_tactical_map

def draw_ball_trajectory(tac_map_copy, ball_track_history):
    """
    Dibuja la trayectoria del balón en el mapa táctico.
    Args:
        tac_map_copy: Mapa táctico
        ball_track_history: Historial de seguimiento del balón
    Returns:
        tac_map_with_trajectory: Mapa táctico con trayectoria
    """
    tac_map_with_trajectory = tac_map_copy.copy()
    if len(ball_track_history['src']) > 0:
        points = np.hstack(ball_track_history['dst']).astype(np.int32).reshape((-1, 1, 2))
        tac_map_with_trajectory = cv2.polylines(tac_map_with_trajectory, [points], isClosed=False,
                                               color=(0, 0, 100), thickness=2)
    return tac_map_with_trajectory

def combine_frames(annotated_frame, tac_map_copy, enable_resize, output_width, output_height):
    """
    Combina el frame anotado y el mapa táctico en la imagen final.
    Args:
        annotated_frame: Frame anotado
        tac_map_copy: Mapa táctico
        enable_resize: Si redimensionar la imagen final
        output_width: Ancho objetivo
        output_height: Alto objetivo
    Returns:
        final_img: Imagen final combinada
    """
    border_color = [0, 0, 0] # Establecer color del borde (BGR)
    annotated_frame = cv2.copyMakeBorder(annotated_frame, 40, 10, 10, 10, # Agregar bordes al frame anotado
                                        cv2.BORDER_CONSTANT, value=border_color)
    tac_map_copy = cv2.copyMakeBorder(tac_map_copy, 70, 50, 10, 10, cv2.BORDER_CONSTANT, # Agregar bordes al mapa táctico
                                    value=border_color)
    tac_map_copy = cv2.resize(tac_map_copy, (tac_map_copy.shape[1], annotated_frame.shape[0])) # Redimensionar mapa táctico
    final_img = cv2.hconcat((annotated_frame, tac_map_copy)) # Concatenar ambas imágenes

    ## Agregar anotación de información
    cv2.putText(final_img, "Mapa Táctico", (1370, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    # Redimensionar final_img si está habilitado
    if enable_resize and output_width and output_height:
        scale = min(output_width / final_img.shape[1], output_height / final_img.shape[0])
        new_width = int(final_img.shape[1] * scale)
        new_height = int(final_img.shape[0] * scale)
        final_img = cv2.resize(final_img, (new_width, new_height))

    return final_img

def add_fps_text(final_img, fps):
    """
    Agrega texto de FPS a la imagen final.
    Args:
        final_img: Imagen final
        fps: Frames por segundo
    Returns:
        final_img: Imagen con texto de FPS
    """
    cv2.putText(final_img, "FPS: " + str(int(fps)), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    return final_img
