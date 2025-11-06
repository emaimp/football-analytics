import os
import json
import yaml
import skimage
from PIL import ImageColor

def get_labels_dics():
    """
    Obtiene el diccionario de posiciones de keypoints del mapa táctico y mapeos de etiquetas.
    Returns:
        keypoints_map_pos: Diccionario de posiciones de keypoints en el mapa táctico
        classes_names_dic: Diccionario que mapea etiquetas numéricas a alfabéticas para keypoints del campo
        labels_dic: Diccionario que mapea etiquetas numéricas a alfabéticas para jugadores/objetos
    """
    # Obtener diccionario de posiciones de keypoints del mapa táctico
    json_path = os.path.join(os.path.dirname(__file__), "../../pitch map labels position.json")
    with open(json_path, 'r') as f:
        keypoints_map_pos = json.load(f)

    # Obtener mapeo numérico a alfabético de keypoints del campo de fútbol
    yaml_path = os.path.join(os.path.dirname(__file__), "../../config pitch dataset.yaml")
    with open(yaml_path, 'r') as file:
        classes_names_dic = yaml.safe_load(file)
    classes_names_dic = classes_names_dic['names']

    # Obtener mapeo numérico a alfabético de keypoints del campo de fútbol
    yaml_path = os.path.join(os.path.dirname(__file__), "../../config players dataset.yaml")
    with open(yaml_path, 'r') as file:
        labels_dic = yaml.safe_load(file)
    labels_dic = labels_dic['names']
    return keypoints_map_pos, classes_names_dic, labels_dic

def create_colors_info(team1_name, team1_p_color, team1_gk_color, team2_name, team2_p_color, team2_gk_color):
    """
    Crea diccionario de información de colores para equipos.
    Args:
        team1_name: Nombre del equipo 1
        team1_p_color: Color de jugadores para el equipo 1
        team1_gk_color: Color de portero para el equipo 1
        team2_name: Nombre del equipo 2
        team2_p_color: Color de jugadores para el equipo 2
        team2_gk_color: Color de portero para el equipo 2
    Returns:
        colors_dic: Diccionario con colores de equipos
        color_list_lab: Lista de colores en espacio Lab para cálculos de distancia
    """
    team1_p_color_rgb = ImageColor.getcolor(team1_p_color, "RGB")
    team1_gk_color_rgb = ImageColor.getcolor(team1_gk_color, "RGB")
    team2_p_color_rgb = ImageColor.getcolor(team2_p_color, "RGB")
    team2_gk_color_rgb = ImageColor.getcolor(team2_gk_color, "RGB")

    colors_dic = {
        team1_name: [team1_p_color_rgb, team1_gk_color_rgb],
        team2_name: [team2_p_color_rgb, team2_gk_color_rgb]
    }
    colors_list = colors_dic[team1_name] + colors_dic[team2_name] # Definir lista de colores para usar en predicción de equipo de jugadores detectados
    color_list_lab = [skimage.color.rgb2lab([i/255 for i in c]) for c in colors_list] # Convirtiendo color_list a espacio L*a*b*
    return colors_dic, color_list_lab

def generate_file_name():
    """
    Genera un nombre de archivo único para videos de salida.
    Returns:
        output_file_name: Cadena de nombre de archivo único
    """
    os.makedirs('./outputs/', exist_ok=True)
    list_video_files = os.listdir('./outputs/')
    idx = 0
    while True:
        idx += 1
        output_file_name = f'detect_{idx}'
        if output_file_name + '.mp4' not in list_video_files:
            break
    return output_file_name
