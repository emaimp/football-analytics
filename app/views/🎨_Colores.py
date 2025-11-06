import cv2
import numpy as np
import skimage.color
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

# Titulo de la página
col_title1, col_title2, col_title3 = st.columns([34, 36, 30])
with col_title2:
    st.header("🎨 Colores")
    st.write("") # Espacio
    st.write("") # Espacio
    st.write("") # Espacio

# Crea diccionario de colores de equipos y lista de colores en espacio Lab
def create_colors_info(team1_name, team1_p_color, team1_gk_color, team2_name, team2_p_color, team2_gk_color):

    # Convierte color hexadecimal a tupla RGB.
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    # Convertir colores hex a RGB
    team1_p_rgb = hex_to_rgb(team1_p_color)
    team1_gk_rgb = hex_to_rgb(team1_gk_color)
    team2_p_rgb = hex_to_rgb(team2_p_color)
    team2_gk_rgb = hex_to_rgb(team2_gk_color)

    # Crear diccionario de colores
    colors_dic = {
        team1_name: [team1_p_rgb, team1_gk_rgb],
        team2_name: [team2_p_rgb, team2_gk_rgb]
    }

    # Convertir colores a espacio Lab
    color_list_lab = []
    for team_colors in colors_dic.values():
        for rgb in team_colors:
            lab = skimage.color.rgb2lab([c/255 for c in rgb])
            color_list_lab.append(lab)

    return colors_dic, color_list_lab

# Configuración de colores de equipo
def render_team_colors(tempf, model_players, team1_name, team2_name, selected_team_info):    
    t1col1, t1col2 = st.columns([1,1])
    with t1col1:
        cap_temp = cv2.VideoCapture(tempf.name)
        frame_count = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_nbr = st.slider(label="Seleccionar fotograma", min_value=1, max_value=frame_count, step=1, help="Seleccionar fotograma para elegir colores de equipo")
        cap_temp.set(cv2.CAP_PROP_POS_FRAMES, frame_nbr)
        success, frame = cap_temp.read()
        with st.spinner('Detectando jugadores en el fotograma seleccionado...'):
            results = model_players(frame, conf=0.7)
            bboxes = results[0].boxes.xyxy.cpu().numpy()
            labels = results[0].boxes.cls.cpu().numpy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections_imgs_list = []
            detections_imgs_grid = []
            padding_img = np.ones((80,60,3),dtype=np.uint8)*255
            for i, j in enumerate(list(labels)):
                if int(j) == 0:
                    bbox = bboxes[i,:]
                    obj_img = frame_rgb[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                    obj_img = cv2.resize(obj_img, (60,80))
                    detections_imgs_list.append(obj_img)
            detections_imgs_grid.append([detections_imgs_list[i] for i in range(len(detections_imgs_list)//2)])
            detections_imgs_grid.append([detections_imgs_list[i] for i in range(len(detections_imgs_list)//2, len(detections_imgs_list))])
            if len(detections_imgs_list)%2 != 0:
                detections_imgs_grid[0].append(padding_img)
            concat_det_imgs_row1 = cv2.hconcat(detections_imgs_grid[0])
            concat_det_imgs_row2 = cv2.hconcat(detections_imgs_grid[1])
            concat_det_imgs = cv2.vconcat([concat_det_imgs_row1,concat_det_imgs_row2])

        st.write("Jugadores detectados")

        value = streamlit_image_coordinates(concat_det_imgs, key="numpy")
        #value_radio_dic = defaultdict(lambda: None)

        st.markdown('---')

        radio_options =[f"Color P de {team1_name}", f"Color GK de {team1_name}", f"Color P de {team2_name}", f"Color GK de {team2_name}"]
        active_color = st.radio(label="Seleccioná qué color de equipo elegir de la imagen", options=radio_options, horizontal=True,
                                help="Elige el color del equipo que quieres seleccionar y haz clic en la imagen de arriba para elegir el color. Los colores se mostrarán en las cajas de abajo.")
        if value is not None:
            picked_color = concat_det_imgs[value['y'], value['x'], :]
            st.session_state[f"{active_color}"] = '#%02x%02x%02x' % tuple(picked_color)

        st.write("Las cajas de abajo se pueden usar para ajustar manualmente los colores seleccionados.")

        cp1, cp2, cp3, cp4 = st.columns([1,1,1,1])
        with cp1:
            hex_color_1 = st.session_state[f"{team1_name} P color"] if f"{team1_name} P color" in st.session_state else selected_team_info["team1_p_color"]
            team1_p_color = st.color_picker(label=' ', value=hex_color_1, key='t1p')
            st.session_state[f"{team1_name} P color"] = team1_p_color
        with cp2:
            hex_color_2 = st.session_state[f"{team1_name} GK color"] if f"{team1_name} GK color" in st.session_state else selected_team_info["team1_gk_color"]
            team1_gk_color = st.color_picker(label=' ', value=hex_color_2, key='t1gk')
            st.session_state[f"{team1_name} GK color"] = team1_gk_color
        with cp3:
            hex_color_3 = st.session_state[f"{team2_name} P color"] if f"{team2_name} P color" in st.session_state else selected_team_info["team2_p_color"]
            team2_p_color = st.color_picker(label=' ', value=hex_color_3, key='t2p')
            st.session_state[f"{team2_name} P color"] = team2_p_color
        with cp4:
            hex_color_4 = st.session_state[f"{team2_name} GK color"] if f"{team2_name} GK color" in st.session_state else selected_team_info["team2_gk_color"]
            team2_gk_color = st.color_picker(label=' ', value=hex_color_4, key='t2gk')
            st.session_state[f"{team2_name} GK color"] = team2_gk_color

    st.markdown('---')

    with t1col2:
        extracted_frame = st.empty()
        extracted_frame.image(frame, use_container_width=True, channels="BGR")

    colors_dic, color_list_lab = create_colors_info(team1_name, st.session_state[f"{team1_name} P color"], st.session_state[f"{team1_name} GK color"],
                                                     team2_name, st.session_state[f"{team2_name} P color"], st.session_state[f"{team2_name} GK color"])
    return colors_dic, color_list_lab

# Ejecutar la configuración de colores
if "input_vide_file" not in st.session_state:
    st.error("Primero carga un video en la pestaña 'Carga de Video'.")
else:
    tempf = st.session_state.tempf
    model_players = st.session_state.model_players
    team1_name = st.session_state.team1_name
    team2_name = st.session_state.team2_name
    selected_team_info = {
        "team1_name": "",
        "team2_name": "",
        "team1_p_color": '#FFFFFF',
        "team1_gk_color": '#000000',
        "team2_p_color": '#FFFFFF',
        "team2_gk_color": '#000000',
    }
    colors_dic, color_list_lab = render_team_colors(tempf, model_players, team1_name, team2_name, selected_team_info)
    st.session_state.colors_dic = colors_dic
    st.session_state.color_list_lab = color_list_lab
