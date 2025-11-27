import cv2
import numpy as np
from sklearn.metrics import mean_squared_error

# Función para calcular la matriz de transformación de homografía cuando se detectan más de 3 keypoints
def calculate_homography(
    detected_labels,
    detected_labels_src_pts,
    detected_labels_dst_pts,
    detected_labels_prev=None,
    detected_labels_src_pts_prev=None,
    k_d_tol=10.0,
    frame_nbr=1
    ):

    homog = None # Inicializar matriz de homografía
    update_homography = False # Inicializar indicador de actualización

    # Si se detectan más de 3 keypoints
    if len(detected_labels) > 3:
        
        # Siempre calcular la matriz de homografía en el primer frame
        # Si no es el primer frame y hay datos previos
        if frame_nbr > 1 and detected_labels_prev is not None and detected_labels_src_pts_prev is not None:
            # Determinar keypoints comunes del campo detectados entre frames anterior y actual
            common_labels = set(detected_labels_prev) & set(detected_labels)
            
            # Cuando se detectan al menos 4 keypoints comunes, determinar si están desplazados en promedio más allá de cierto nivel de tolerancia
            # Si hay al menos 4 keypoints comunes
            if len(common_labels) > 3:
                # Obtener índices de etiquetas de keypoints comunes detectados del frame anterior
                common_label_idx_prev = [detected_labels_prev.index(i) for i in common_labels]
                # Obtener índices de etiquetas de keypoints comunes detectados del frame actual
                common_label_idx_curr = [detected_labels.index(i) for i in common_labels]
                # Obtener coordenadas de keypoints comunes detectados del frame anterior
                coor_common_label_prev = detected_labels_src_pts_prev[common_label_idx_prev]
                # Obtener coordenadas de keypoints comunes detectados del frame actual
                coor_common_label_curr = detected_labels_src_pts[common_label_idx_curr]
                # Calcular error entre coordenadas de keypoints comunes anteriores y actuales
                coor_error = mean_squared_error(coor_common_label_prev, coor_common_label_curr)
                # Verificar si el error superó el nivel de tolerancia predefinido
                update_homography = coor_error > k_d_tol
            else:
                update_homography = True # Actualizar si no hay suficientes comunes
        else:
            update_homography = True # Actualizar en primer frame

        # Si se debe actualizar
        if update_homography:
            homog, mask = cv2.findHomography(detected_labels_src_pts, detected_labels_dst_pts) # Calcular homografía

    # Actualizar datos del frame anterior
    if homog is not None:
        detected_labels_prev = detected_labels.copy() # Copiar etiquetas actuales
        detected_labels_src_pts_prev = detected_labels_src_pts.copy() # Copiar puntos actuales

    return homog, update_homography, detected_labels_prev, detected_labels_src_pts_prev # Retornar resultados

# Función para transformar puntos del plano fuente al plano destino usando la matriz de homografía
def transform_points(homog, points):

    transformed_points = [] # Inicializar lista de puntos transformados

    # Bucle sobre cada punto a transformar
    for pt in points:
        pt_homog = np.append(np.array(pt), np.array([1]), axis=0) # Convertir a coordenadas homogéneas
        dest_point = np.matmul(homog, np.transpose(pt_homog)) # Aplicar transformación de homografía
        dest_point = dest_point / dest_point[2] # Revertir a coordenadas 2D
        transformed_points.append(list(np.transpose(dest_point)[:2])) # Agregar punto transformado
    return np.array(transformed_points) # Retornar array de puntos transformados

# Función para actualizar el historial de seguimiento del balón
def update_ball_tracking(
    ball_track_history,
    detected_ball_src_pos,
    detected_ball_dst_pos,
    ball_track_dist_thresh,
    max_track_length
    ):

    # Si se detectó el balón
    if detected_ball_src_pos is not None and detected_ball_dst_pos is not None:
        
        # Si hay historial previo
        if len(ball_track_history['src']) > 0:
            
            # Si está cerca del último punto
            if np.linalg.norm(detected_ball_src_pos - ball_track_history['src'][-1]) < ball_track_dist_thresh:
                ball_track_history['src'].append((int(detected_ball_src_pos[0]), int(detected_ball_src_pos[1]))) # Agregar posición fuente
                ball_track_history['dst'].append((int(detected_ball_dst_pos[0]), int(detected_ball_dst_pos[1]))) # Agregar posición destino
            else: # Si no está cerca, reiniciar seguimiento
                ball_track_history['src'] = [(int(detected_ball_src_pos[0]), int(detected_ball_src_pos[1]))] # Reiniciar fuente
                ball_track_history['dst'] = [(int(detected_ball_dst_pos[0]), int(detected_ball_dst_pos[1]))] # Reiniciar destino
        else: # Si no hay historial, agregar primera posición
            ball_track_history['src'].append((int(detected_ball_src_pos[0]), int(detected_ball_src_pos[1]))) # Agregar primera fuente
            ball_track_history['dst'].append((int(detected_ball_dst_pos[0]), int(detected_ball_dst_pos[1]))) # Agregar primera destino

    # Limitar longitud del seguimiento
    if len(ball_track_history['src']) > max_track_length: # Si excede la longitud máxima
        ball_track_history['src'].pop(0) # Remover el más antiguo en fuente
        ball_track_history['dst'].pop(0) # Remover el más antiguo en destino

    return ball_track_history
