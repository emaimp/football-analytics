import cv2
import numpy as np

# Función para anotar el frame con bounding boxes, colores de equipo y etiquetas de texto
def annotate_frame(
    frame, bboxes_p,
    labels_p,
    confs_p,
    players_teams_list,
    colors_dic,
    obj_palette_list,
    labels_dic,
    show_pal,
    show_p,
    show_k,
    bboxes_k,
    player_ids=None,
    show_b=False
    ):

    annotated_frame = frame.copy() # Copiar frame original
    palette_box_size = 10 # Tamaño de la caja de color en píxeles (para visualización)
    j = 0 # Inicializando contador de jugadores detectados

    # Bucle sobre todos los objetos detectados
    for i in range(bboxes_p.shape[0]):
        conf = confs_p[i] # Obtener confianza del objeto detectado actual

        # Mostrar anotación para jugadores detectados (etiqueta 0)
        if labels_p[i] == 0:

            # Mostrar paleta de colores extraída para cada jugador detectado
            if show_pal and j < len(obj_palette_list):
                palette = obj_palette_list[j] # Obtener paleta de colores del jugador detectado

                for k, c in enumerate(palette):
                    c_bgr = c[::-1] # Convertir color a BGR
                    # Agregar anotación de paleta de colores en el frame
                    annotated_frame = cv2.rectangle(
                        annotated_frame,
                        (int(bboxes_p[i, 2]) + 3,
                        int(bboxes_p[i, 1]) + k * palette_box_size),
                        (int(bboxes_p[i, 2]) + palette_box_size,
                        int(bboxes_p[i, 1]) + (palette_box_size) * (k + 1)),
                        c_bgr, -1)

            team_name = list(colors_dic.keys())[players_teams_list[j]] # Obtener predicción de equipo del jugador detectado
            color_rgb = colors_dic[team_name][0] # Obtener color de equipo del jugador detectado
            color_bgr = color_rgb[::-1] # Convertir color a bgr

            if show_p:
                # Agregar anotaciones de bbox con colores de equipo
                annotated_frame = cv2.rectangle(
                    annotated_frame,
                    (int(bboxes_p[i, 0]),
                    int(bboxes_p[i, 1])),
                    (int(bboxes_p[i, 2]),
                    int(bboxes_p[i, 3])),
                    color_bgr, 1)

                # Usar ID secuencial si está disponible, sino nombre de equipo + confianza
                if player_ids is not None and j < len(player_ids):
                    text = str(player_ids[j])
                else:
                    text = team_name + f" {conf:.2f}"

                # Agregar anotaciones
                annotated_frame = cv2.putText(
                    annotated_frame,
                    text,
                    (int(bboxes_p[i, 0]),
                    int(bboxes_p[i, 1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color_bgr, 2
                    )

            j += 1 # Actualizar contador de jugadores

        # Mostrar anotación para árbitros y balón solo si show_b
        elif (labels_p[i] == 1) or (labels_p[i] == 2 and show_b):
            # Agregar anotaciones de bbox de color blanco
            annotated_frame = cv2.rectangle(
                annotated_frame,
                (int(bboxes_p[i, 0]),
                int(bboxes_p[i, 1])),
                (int(bboxes_p[i, 2]),
                int(bboxes_p[i, 3])),
                (255, 255, 255), 1
                )
            # Agregar anotaciones de texto de etiqueta de color blanco
            text = labels_dic[labels_p[i]] if labels_p[i] == 1 else labels_dic[labels_p[i]] + f" {conf:.2f}"
            annotated_frame = cv2.putText(
                annotated_frame,
                text,
                (int(bboxes_p[i, 0]), int(bboxes_p[i, 1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 2
                )

    # Anotar keypoints si está habilitado
    if show_k:
        for i in range(bboxes_k.shape[0]):
            # Agregar anotaciones de bbox con colores de equipo
            annotated_frame = cv2.rectangle(
                annotated_frame,
                (int(bboxes_k[i, 0]),
                int(bboxes_k[i, 1])),
                (int(bboxes_k[i, 2]),
                int(bboxes_k[i, 3])),
                (0, 0, 0), 1
                )
    return annotated_frame

# Función para anotar el mapa táctico con posiciones de jugadores y balón
def annotate_tactical_map(
    tac_map_copy,
    pred_dst_pts,
    detected_ball_dst_pos,
    players_teams_list,
    colors_dic,
    player_ids=None,
    show_b=False
    ):

    annotated_tactical_map = tac_map_copy.copy() # Copiar mapa táctico
    ball_color_bgr = (0, 0, 255) # Color (BGR) para anotación del balón en el mapa táctico

    # Anotar posiciones de jugadores
    if pred_dst_pts is not None:
        for j, pt in enumerate(pred_dst_pts): # Bucle sobre posiciones de jugadores
            team_name = list(colors_dic.keys())[players_teams_list[j]] # Obtener nombre de equipo
            color_rgb = colors_dic[team_name][0] # Obtener color RGB
            color_bgr = color_rgb[::-1] # Convertir a BGR

            # Dibujar círculo con borde blanco y relleno azul
            center = (int(pt[0]), int(pt[1])) # Centro del círculo
            radius = 10 # Radio del círculo

            # Borde blanco con antialiasing
            annotated_tactical_map = cv2.circle(
                annotated_tactical_map,
                center,
                radius + 2,
                (255, 255, 255),
                thickness=2,
                lineType=cv2.LINE_AA
            )

            # Relleno azul con antialiasing
            annotated_tactical_map = cv2.circle(
                annotated_tactical_map,
                center,
                radius,
                (0, 0, 255),
                thickness=-1,
                lineType=cv2.LINE_AA
            )

            # Agregar ID del jugador si está disponible
            if player_ids is not None and j < len(player_ids): # Si hay IDs de jugadores
                annotated_tactical_map = cv2.putText(
                    annotated_tactical_map,
                    str(player_ids[j]),
                    (int(pt[0]) + 12,
                    int(pt[1]) - 12),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6,
                    (0, 0, 0), 1,
                    lineType=cv2.LINE_AA
                )

    # Anotar posición del balón solo si está habilitado
    if detected_ball_dst_pos is not None and show_b: # Si balón detectado y habilitado
        annotated_tactical_map = cv2.circle(
            annotated_tactical_map,
            (int(detected_ball_dst_pos[0]),
            int(detected_ball_dst_pos[1])), radius=5,
            color=ball_color_bgr, thickness=3) # Dibujar balón
    return annotated_tactical_map

# Función para dibujar la trayectoria del balón en el mapa táctico
def draw_ball_trajectory(tac_map_copy, ball_track_history):

    tac_map_with_trajectory = tac_map_copy.copy()# Copiar mapa táctico
    if len(ball_track_history['src']) > 0: # Si hay historial de balón
        points = np.hstack(ball_track_history['dst']).astype(np.int32).reshape((-1, 1, 2)) # Preparar puntos para polilínea
        tac_map_with_trajectory = cv2.polylines(
            tac_map_with_trajectory,
            [points],
            isClosed=False,
            color=(0, 0, 100),
            thickness=2
            ) # Dibujar trayectoria
    return tac_map_with_trajectory

# Función para combinar el frame anotado y el mapa táctico en la imagen final
def combine_frames(annotated_frame, tac_map_copy):
    """
    Combina el frame anotado y el mapa táctico en la imagen final.
    Args:
        annotated_frame: Frame anotado
        tac_map_copy: Mapa táctico
    Returns:
        final_img: Imagen final combinada
    """
    border_color_game = [0, 0, 255] # Establecer color del borde para video del juego (BGR)
    border_color_tactical = [0, 0, 0] # Establecer color del borde para mapa táctico (BGR)

    # Agregar bordes gruesos al frame anotado
    annotated_frame = cv2.copyMakeBorder(
        annotated_frame, 3, 3, 3, 3,
        cv2.BORDER_CONSTANT,
        value=border_color_game
    )# Agregar borde al frame

    tac_map_copy = cv2.copyMakeBorder(
        tac_map_copy, 50, 50, 10, 10,
        cv2.BORDER_CONSTANT,
        value=border_color_tactical
    ) # Agregar borde al mapa

    tac_map_copy = cv2.resize(tac_map_copy, (tac_map_copy.shape[1], annotated_frame.shape[0])) # Redimensionar mapa
    final_img = cv2.hconcat((annotated_frame, tac_map_copy)) # Concatenar horizontalmente
    return final_img

# Función para agregar texto de FPS a la imagen final
def add_fps_text(final_img, fps):

    cv2.putText(final_img, "FPS: " + str(int(fps)), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2) # Agregar texto de FPS
    return final_img

# Función para agregar texto al mapa táctico
def add_fps_text_tactical(final_img):

    cv2.putText(final_img, "Mapa Tactico", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2) # Agregar texto al mapa
    return final_img
