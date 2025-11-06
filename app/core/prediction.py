import cv2
import numpy as np
import pandas as pd
import skimage.color
from PIL import Image

def extract_player_palettes(frame, bboxes_p, labels_p, num_pal_colors):
    """
    Extrae paletas de colores dominantes de jugadores detectados.
    Args:
        frame: Frame de entrada (BGR)
        bboxes_p: Bounding boxes de jugadores (xyxy)
        labels_p: Etiquetas de detección
        num_pal_colors: Número de colores a extraer de la paleta
    Returns:
        obj_palette_list: Lista de paletas de colores para cada jugador
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convertir frame a RGB
    obj_palette_list = [] # Inicializar lista de paletas de colores de jugadores
    palette_interval = (0, num_pal_colors) # Intervalo de colores a extraer de la paleta de colores dominantes

    # Bucle sobre jugadores detectados (etiqueta 0) y extraer paleta de colores dominantes basada en intervalo definido
    for i, j in enumerate(labels_p):
        if int(j) == 0:
            bbox = bboxes_p[i, :]  # Obtener info de bbox (x,y,x,y)
            obj_img = frame_rgb[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]  # Recortar bbox del frame
            obj_img_w, obj_img_h = obj_img.shape[1], obj_img.shape[0]
            center_filter_x1 = np.max([(obj_img_w//2)-(obj_img_w//5), 1])
            center_filter_x2 = (obj_img_w//2)+(obj_img_w//5)
            center_filter_y1 = np.max([(obj_img_h//3)-(obj_img_h//5), 1])
            center_filter_y2 = (obj_img_h//3)+(obj_img_h//5)
            center_filter = obj_img[center_filter_y1:center_filter_y2,
                                    center_filter_x1:center_filter_x2]
            obj_pil_img = Image.fromarray(np.uint8(center_filter)) # Convertir a imagen pillow
            reduced = obj_pil_img.convert("P", palette=Image.Palette.WEB) # Convertir a paleta web (216 colores)
            palette = reduced.getpalette() # Obtener paleta como [r,g,b,r,g,b,...]
            palette = [palette[3*n:3*n+3] for n in range(256)] # Agrupar 3 por 3 = [[r,g,b],[r,g,b],...]
            color_count = [(n, palette[m]) for n, m in reduced.getcolors()] # Crear lista de colores de paleta con su frecuencia
            RGB_df = pd.DataFrame(color_count, columns=['cnt', 'RGB']).sort_values( # Crear dataframe basado en intervalo de paleta definido
                            by='cnt', ascending=False).iloc[
                                palette_interval[0]:palette_interval[1], :]
            palette = list(RGB_df.RGB) # Convertir paleta a lista (para procesamiento más rápido)

            # Actualizar lista de paletas de colores de jugadores detectados
            obj_palette_list.append(palette)

    return obj_palette_list

def calculate_distance_features(obj_palette_list, color_list_lab):
    """
    Calcula distancias entre paletas de jugadores y colores de equipos.
    Args:
        obj_palette_list: Lista de paletas de colores de jugadores
        color_list_lab: Colores de equipos en espacio Lab
    Returns:
        players_distance_features: Características de distancia para cada jugador
    """
    players_distance_features = []
    # Bucle sobre paletas de colores extraídas de jugadores detectados
    for palette in obj_palette_list:
        palette_distance = []
        palette_lab = [skimage.color.rgb2lab([i/255 for i in color]) for color in palette]  # Convertir colores a espacio L*a*b*
        # Bucle sobre colores en paleta
        for color in palette_lab:
            distance_list = []
            # Bucle sobre lista predefinida de colores de equipos
            for c in color_list_lab:
                distance = skimage.color.deltaE_cie76(color, c) # Calcular distancia euclidiana en espacio de color Lab
                distance_list.append(distance) # Actualizar lista de distancias para color actual
            palette_distance.append(distance_list) # Actualizar lista de distancias para paleta actual
        players_distance_features.append(palette_distance) # Actualizar lista de características de distancia

    return players_distance_features

def predict_teams(distance_features, nbr_team_colors):
    """
    Predice equipos de jugadores basados en características de distancia.
    Args:
        distance_features: Características de distancia para cada jugador
        nbr_team_colors: Número de colores por equipo
    Returns:
        players_teams_list: Índices de equipos predichos para cada jugador
    """
    players_teams_list = []
    # Bucle sobre características de distancia de jugadores
    for distance_feats in distance_features:
        vote_list = []
        # Bucle sobre distancias para cada color
        for dist_list in distance_feats:
            team_idx = dist_list.index(min(dist_list)) // nbr_team_colors # Asignar índice de equipo para color actual basado en distancia mínima
            vote_list.append(team_idx) # Actualizar lista de votos con predicción de equipo de color actual
        players_teams_list.append(max(vote_list, key=vote_list.count)) # Predecir equipo de jugador actual por conteo de votos

    return players_teams_list
