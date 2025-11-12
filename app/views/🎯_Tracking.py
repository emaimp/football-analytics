import cv2
import json
import numpy as np
import streamlit as st
from core import homography
import matplotlib.pyplot as plt

# Titulo de la página
col_title1, col_title2, col_title3 = st.columns([40, 35, 25])
with col_title2:
    st.header("🎯 Tracking")
    st.write("") # Espacio
    st.write("") # Espacio
    st.write("") # Espacio

# Función principal: Detecta jugadores en el primer frame y muestra mapa táctico
def render_team_colors(tempf, model_players, model_keypoints):
    cap_temp = cv2.VideoCapture(tempf.name)
    cap_temp.set(cv2.CAP_PROP_POS_FRAMES, 1) # Mostrar solo el primer frame
    frame = cap_temp.read()[1] # Leer frame (ignorar success)
    frame_original = frame.copy() # Copia para mostrar sin anotaciones

    with st.spinner('Detectando jugadores...'):
        # Ejecutar modelo de tracking en el frame
        results = model_players.track(frame, conf=0.4, persist=True, tracker="botsort.yaml")
        # Extraer bounding boxes, etiquetas y IDs de detección
        bboxes = results[0].boxes.xyxy.cpu().numpy() # Coordenadas de cajas
        labels = results[0].boxes.cls.cpu().numpy() # Clases detectadas
        ids = results[0].boxes.id # IDs de tracking

        if ids is not None:
            ids = ids.cpu().numpy() # Convertir a numpy si existen
        else:
            ids = np.arange(len(bboxes)) # IDs fallback si no hay tracking

        # Lista de colores disponibles para asignar a jugadores
        player_colors_list = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'cyan', 'magenta', 'brown']
        # Diccionario para convertir nombres de colores a BGR (OpenCV)
        color_dict = {
            'red': (0, 0, 255),
            'blue': (255, 0, 0),
            'green': (0, 255, 0),
            'yellow': (0, 255, 255),
            'purple': (255, 0, 255),
            'orange': (0, 165, 255),
            'pink': (203, 192, 255),
            'cyan': (255, 255, 0),
            'magenta': (255, 0, 255),
            'brown': (42, 42, 165)
        }
        # Listas para almacenar datos de jugadores
        player_centers = [] # Centros de bounding boxes
        player_colors = [] # Colores asignados
        player_ids = [] # IDs de jugadores

        # Bucle para procesar cada detección
        for i, j in enumerate(list(labels)):
            if int(j) == 0: # Procesar solo clase 0 (jugadores)
                bbox = bboxes[i,:] # Bounding box actual
                x1, y1, x2, y2 = bbox.astype(int) # Coordenadas enteras
                center = [(x1 + x2)/2, (y1 + y2)/2] # Centro del bbox
                # Asignar color único basado en ID (cíclico)
                player_color = player_colors_list[int(ids[i]) % len(player_colors_list)]
                color_draw = color_dict[player_color] # Color BGR para dibujo
                # Almacenar datos
                player_centers.append(center)
                player_colors.append(player_color)
                player_ids.append(int(ids[i]))
                # Dibujar rectángulo y texto en el frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_draw, 2)
                cv2.putText(frame, f"{int(ids[i])}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_draw, 2)

        # Detectar keypoints del campo para transformación homográfica
        results_keypoints = model_keypoints(frame, conf=0.7)
        bboxes_keypoints = results_keypoints[0].boxes.xyxy.cpu().numpy() # Bboxes de keypoints
        labels_keypoints = results_keypoints[0].boxes.cls.cpu().numpy() # Clases de keypoints
        # Mapeo de clases a nombres de puntos del campo
        names = {0: 'TLC', 1: 'TRC', 2: 'TR6MC', 3: 'TL6MC', 4: 'TR6ML', 5: 'TL6ML', 6: 'TR18MC', 7: 'TL18MC', 8: 'TR18ML', 9: 'TL18ML', 10: 'TRArc', 11: 'TLArc', 12: 'RML', 13: 'RMC', 14: 'LMC', 15: 'LML', 16: 'BLC', 17: 'BRC', 18: 'BR6MC', 19: 'BL6MC', 20: 'BR6ML', 21: 'BL6ML', 22: 'BR18MC', 23: 'BL18MC', 24: 'BR18ML', 25: 'BL18ML', 26: 'BRArc', 27: 'BLArc'}
        detected_labels = [names[int(cls)] for cls in labels_keypoints] # Nombres detectados
        detected_src_pts = [] # Puntos fuente detectados

        # Calcular centros de bboxes de keypoints
        for bbox in bboxes_keypoints:
            center = [(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2]
            detected_src_pts.append(center)
        detected_src_pts = np.array(detected_src_pts)

        # Cargar posiciones predefinidas del mapa táctico
        with open('app/config/map_labels_position.json') as f:
            positions = json.load(f)

        # Filtrar keypoints comunes entre detectados y mapa
        common_labels = [label for label in detected_labels if label in positions]
        detected_src_pts_filtered = np.array([detected_src_pts[detected_labels.index(label)] for label in common_labels])
        detected_dst_pts = np.array([positions[label] for label in common_labels])

        # Calcular matriz de homografía para transformar coordenadas
        h_matrix, _, _, _ = homography.calculate_homography(common_labels, detected_src_pts_filtered, detected_dst_pts, 1)

        # Transformar posiciones de jugadores al plano del mapa
        transformed_centers = None
        if h_matrix is not None and player_centers:
            transformed_centers = homography.transform_points(h_matrix, player_centers)

    st.write("Primer Fotograma.") # Subtítulo

    # Mostrar frames en dos columnas
    col1, col2 = st.columns(2)
    with col1:
        til1, til2, til3 = st.columns([25, 50, 25])
        with til2:
            st.subheader("Imagen - Original") # Título
        st.image(frame_original, channels="BGR", use_container_width=True) # Mostrar frame sin anotaciones

    with col2:
        til1, til2, til3 = st.columns([25, 60, 15])
        with til2:
            st.subheader("Detección - Jugadores") # Título
        st.image(frame, channels="BGR", use_container_width=True) # Mostrar frame con rectángulos y IDs

    st.write("") # Espacio
    st.write("") # Espacio
    st.write("Campo Táctico.") # Subtítulo

    # Mostrar mapas tácticos en dos columnas
    col_map1, col_map2 = st.columns(2)
    with col_map1:
        cen1, cen2, cen3 = st.columns([20, 70, 10])
        with cen2:
            til1, til2, til3 = st.columns([19, 50, 31])
            with til2:
                st.subheader("Coordenadas") # Título
            with open('app/config/map_labels_position.json') as f:
                positions = json.load(f) # Cargar posiciones de puntos del campo
            img = plt.imread('app/assets/campo_tactico.png') # Cargar imagen del campo
            fig, ax = plt.subplots()
            fig.set_size_inches(6, 4)
            fig.patch.set_facecolor('black')  # Fondo negro
            ax.imshow(img) # Mostrar imagen de fondo
            # Dibujar puntos de referencia del campo
            for label, (x, y) in positions.items():
                ax.plot(x, y, 'ro', markersize=3)
                ax.text(x, y, label, fontsize=5, ha='center', va='bottom', color='white', fontweight='bold')
            ax.axis('off')
            st.pyplot(fig, width=400)

    with col_map2:
        cen1, cen2, cen3 = st.columns([20, 70, 10])
        with cen2:
            til1, til2, til3 = st.columns([3, 90, 7])
            with til2:
                st.subheader("Detección - Jugadores") # Título
            img_clean = plt.imread('app/assets/campo_tactico.png') # Cargar imagen limpia del campo
            fig, ax = plt.subplots()
            fig.set_size_inches(6, 4)
            fig.patch.set_facecolor('black')  # Fondo negro
            ax.imshow(img_clean)
            if transformed_centers is not None: # Si hay posiciones transformadas
                # Dibujar puntos de jugadores con colores únicos
                for i, (x, y) in enumerate(transformed_centers):
                    ax.scatter(x, y, facecolors=player_colors[i], edgecolors=player_colors[i], s=64)
                    ax.text(x, y, f"{player_ids[i]}", fontsize=8, ha='center', va='bottom', color='black', fontweight='bold')
            ax.axis('off')
            st.pyplot(fig, width=400)

    # Retornar diccionario de colores fijo
    colors_dic = {"Jugadores": [(0, 0, 0), (0, 0, 0)]} # Azul para jugadores, negro para GK (no usado)
    return colors_dic

# Ejecutar la configuración de colores
if "input_vide_file" not in st.session_state: # Verificar si hay video cargado
    st.error("Primero carga un video en la pestaña 'Carga de Video'.")
else:
    tempf = st.session_state.tempf # Obtener el archivo temporal del video
    model_players = st.session_state.model_players # Obtener el modelo de detección de jugadores
    colors_dic = render_team_colors(tempf, model_players, st.session_state.model_keypoints) # Ejecutar detección y obtener colores
    st.session_state.colors_dic = colors_dic # Guardar diccionario de colores en estado
