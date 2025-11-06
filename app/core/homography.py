import cv2
import numpy as np
from sklearn.metrics import mean_squared_error

def calculate_homography(detected_labels, detected_labels_src_pts, detected_labels_dst_pts,
                        detected_labels_prev=None, detected_labels_src_pts_prev=None, k_d_tol=10.0, frame_nbr=1):
    """
    Calcula la matriz de transformación de homografía cuando se detectan más de 3 keypoints.
    Args:
        detected_labels: Lista de etiquetas de keypoints detectados para el frame actual
        detected_labels_src_pts: Puntos fuente (coordenadas del frame) para el frame actual
        detected_labels_dst_pts: Puntos destino (coordenadas del mapa)
        detected_labels_prev: Etiquetas del frame anterior
        detected_labels_src_pts_prev: Puntos fuente del frame anterior
        k_d_tol: Tolerancia para desplazamiento de keypoints
        frame_nbr: Número del frame actual
    Returns:
        homog: Matriz de homografía si es calculable, None en caso contrario
        update_homography: Booleano que indica si la homografía fue actualizada
        detected_labels_prev: Etiquetas anteriores actualizadas
        detected_labels_src_pts_prev: Puntos anteriores actualizados
    """
    homog = None
    update_homography = False

    if len(detected_labels) > 3:
        # Siempre calcular la matriz de homografía en el primer frame
        if frame_nbr > 1 and detected_labels_prev is not None and detected_labels_src_pts_prev is not None:
            # Determinar keypoints comunes del campo detectados entre frames anterior y actual
            common_labels = set(detected_labels_prev) & set(detected_labels)
            # Cuando se detectan al menos 4 keypoints comunes, determinar si están desplazados en promedio más allá de cierto nivel de tolerancia
            if len(common_labels) > 3:
                common_label_idx_prev = [detected_labels_prev.index(i) for i in common_labels] # Obtener índices de etiquetas de keypoints comunes detectados del frame anterior
                common_label_idx_curr = [detected_labels.index(i) for i in common_labels] # Obtener índices de etiquetas de keypoints comunes detectados del frame actual
                coor_common_label_prev = detected_labels_src_pts_prev[common_label_idx_prev] # Obtener coordenadas de keypoints comunes detectados del frame anterior
                coor_common_label_curr = detected_labels_src_pts[common_label_idx_curr] # Obtener coordenadas de keypoints comunes detectados del frame actual
                coor_error = mean_squared_error(coor_common_label_prev, coor_common_label_curr) # Calcular error entre coordenadas de keypoints comunes anteriores y actuales
                update_homography = coor_error > k_d_tol # Verificar si el error superó el nivel de tolerancia predefinido
            else:
                update_homography = True
        else:
            update_homography = True

        if update_homography:
            homog, mask = cv2.findHomography(detected_labels_src_pts, detected_labels_dst_pts)

    # Actualizar datos del frame anterior
    if homog is not None:
        detected_labels_prev = detected_labels.copy()
        detected_labels_src_pts_prev = detected_labels_src_pts.copy()

    return homog, update_homography, detected_labels_prev, detected_labels_src_pts_prev

def transform_points(homog, points):
    """
    Transforma puntos del plano fuente al plano destino usando la matriz de homografía.
    Args:
        homog: Matriz de homografía
        points: Lista de puntos a transformar (x, y)
    Returns:
        transformed_points: Lista de puntos transformados
    """
    transformed_points = []
    for pt in points:
        pt_homog = np.append(np.array(pt), np.array([1]), axis=0) # Convertir a coordenadas homogéneas
        dest_point = np.matmul(homog, np.transpose(pt_homog)) # Aplicar transformación de homografía
        dest_point = dest_point / dest_point[2] # Revertir a coordenadas 2D
        transformed_points.append(list(np.transpose(dest_point)[:2]))
    return np.array(transformed_points)

def update_ball_tracking(ball_track_history, detected_ball_src_pos, detected_ball_dst_pos,
                        ball_track_dist_thresh, max_track_length):
    """
    Actualiza el historial de seguimiento del balón.
    Args:
        ball_track_history: Diccionario con listas 'src' y 'dst'
        detected_ball_src_pos: Posición del balón en el frame fuente
        detected_ball_dst_pos: Posición del balón en el mapa destino
        ball_track_dist_thresh: Umbral de distancia para seguimiento
        max_track_length: Longitud máxima del historial de seguimiento
    Returns:
        ball_track_history actualizado
    """
    if detected_ball_src_pos is not None and detected_ball_dst_pos is not None:
        if len(ball_track_history['src']) > 0:
            if np.linalg.norm(detected_ball_src_pos - ball_track_history['src'][-1]) < ball_track_dist_thresh:
                ball_track_history['src'].append((int(detected_ball_src_pos[0]), int(detected_ball_src_pos[1])))
                ball_track_history['dst'].append((int(detected_ball_dst_pos[0]), int(detected_ball_dst_pos[1])))
            else:
                ball_track_history['src'] = [(int(detected_ball_src_pos[0]), int(detected_ball_src_pos[1]))]
                ball_track_history['dst'] = [(int(detected_ball_dst_pos[0]), int(detected_ball_dst_pos[1]))]
        else:
            ball_track_history['src'].append((int(detected_ball_src_pos[0]), int(detected_ball_src_pos[1])))
            ball_track_history['dst'].append((int(detected_ball_dst_pos[0]), int(detected_ball_dst_pos[1])))

    # Limitar longitud del seguimiento
    if len(ball_track_history['src']) > max_track_length:
        ball_track_history['src'].pop(0)
        ball_track_history['dst'].pop(0)

    return ball_track_history
