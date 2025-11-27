import cv2
import numpy as np
import pandas as pd
import skimage.color
from PIL import Image

# Función para extraer paletas de colores dominantes de jugadores detectados
def extract_player_palettes(frame, bboxes_p, labels_p, num_pal_colors):

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convertir frame a RGB
    obj_palette_list = [] # Inicializar lista de paletas de colores de jugadores
    palette_interval = (0, num_pal_colors) # Intervalo de colores a extraer de la paleta de colores dominantes

    # Bucle sobre jugadores detectados (etiqueta 0) y extraer paleta de colores dominantes basada en intervalo definido
    for i, j in enumerate(labels_p):
        if int(j) == 0:
            bbox = bboxes_p[i, :] # Obtener info de bbox (x,y,x,y)
            obj_img = frame_rgb[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] # Recortar bbox del frame
            obj_img_w, obj_img_h = obj_img.shape[1], obj_img.shape[0] # Ancho y alto de la imagen del objeto
            center_filter_x1 = np.max([(obj_img_w//2)-(obj_img_w//5), 1]) # Coordenada x1 del filtro central
            center_filter_x2 = (obj_img_w//2)+(obj_img_w//5) # Coordenada x2 del filtro central
            center_filter_y1 = np.max([(obj_img_h//3)-(obj_img_h//5), 1]) # Coordenada y1 del filtro central
            center_filter_y2 = (obj_img_h//3)+(obj_img_h//5) # Coordenada y2 del filtro central
            center_filter = obj_img[center_filter_y1:center_filter_y2, center_filter_x1:center_filter_x2] # Aplicar filtro central

            obj_pil_img = Image.fromarray(np.uint8(center_filter)) # Convertir a imagen pillow
            reduced = obj_pil_img.convert("P", palette=Image.Palette.WEB) # Convertir a paleta web (216 colores)
            palette = reduced.getpalette() # Obtener paleta como [r,g,b,r,g,b,...]
            palette = [palette[3*n:3*n+3] for n in range(256)] # Agrupar 3 por 3 = [[r,g,b],[r,g,b],...]
            color_count = [(n, palette[m]) for n, m in reduced.getcolors()] # Crear lista de colores de paleta con su frecuencia
            
            # Crear dataframe basado en intervalo de paleta definido
            RGB_df = pd.DataFrame(color_count, columns=['cnt', 'RGB']).sort_values(by='cnt', ascending=False).iloc[palette_interval[0]:palette_interval[1], :]
            palette = list(RGB_df.RGB) # Convertir paleta a lista (para procesamiento más rápido)
            obj_palette_list.append(palette) # Actualizar lista de paletas de colores de jugadores detectados

    return obj_palette_list

# Función para calcular distancias entre paletas de jugadores y colores de equipos
def calculate_distance_features(obj_palette_list, color_list_lab):

    players_distance_features = [] # Inicializar lista de características de distancia

    # Bucle sobre paletas de colores extraídas de jugadores detectados
    for palette in obj_palette_list:
        palette_distance = [] # Inicializar lista de distancias para la paleta actual
        palette_lab = [skimage.color.rgb2lab([i/255 for i in color]) for color in palette] # Convertir colores a espacio L*a*b*
        
        # Bucle sobre colores en paleta
        for color in palette_lab:
            distance_list = [] # Inicializar lista de distancias para el color actual
            
            # Bucle sobre lista predefinida de colores de equipos
            for c in color_list_lab:
                distance = skimage.color.deltaE_cie76(color, c) # Calcular distancia euclidiana en espacio de color Lab
                distance_list.append(distance) # Actualizar lista de distancias para color actual
            palette_distance.append(distance_list) # Actualizar lista de distancias para paleta actual
        players_distance_features.append(palette_distance) # Actualizar lista de características de distancia

    return players_distance_features # Retornar características de distancia

# Función para predecir equipos de jugadores basados en características de distancia
def predict_teams(distance_features, nbr_team_colors):

    players_teams_list = [] # Inicializar lista de equipos predichos

    # Bucle sobre características de distancia de jugadores
    for distance_feats in distance_features:
        vote_list = [] # Inicializar lista de votos para el jugador actual
        
        # Bucle sobre distancias para cada color
        for dist_list in distance_feats:
            team_idx = dist_list.index(min(dist_list)) // nbr_team_colors # Asignar índice de equipo para color actual basado en distancia mínima
            vote_list.append(team_idx) # Actualizar lista de votos con predicción de equipo de color actual
        players_teams_list.append(max(vote_list, key=vote_list.count)) # Predecir equipo de jugador actual por conteo de votos

    return players_teams_list # Retornar lista de equipos predichos
