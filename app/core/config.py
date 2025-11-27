import os
import json
import yaml
import skimage
from PIL import ImageColor

# Función para obtener diccionarios de posiciones de keypoints y mapeos de etiquetas
def get_labels_dics():

    # Obtener diccionario de posiciones de keypoints del mapa táctico
    json_path = os.path.join(os.path.dirname(__file__), "../config/map_labels_position.json")
    # Abrir archivo JSON
    with open(json_path, 'r') as f:
        keypoints_map_pos = json.load(f) # Cargar posiciones de keypoints

    # Obtener mapeo numérico a alfabético de keypoints del campo de fútbol
    yaml_path = os.path.join(os.path.dirname(__file__), "../config/pitch_dataset.yaml")
    # Abrir archivo YAML
    with open(yaml_path, 'r') as file:
        classes_names_dic = yaml.safe_load(file) # Cargar diccionario de clases
    classes_names_dic = classes_names_dic['names'] # Extraer nombres de clases

    # Obtener mapeo numérico a alfabético de jugadores/objetos
    yaml_path = os.path.join(os.path.dirname(__file__), "../config/players_dataset.yaml")
    # Abrir archivo YAML
    with open(yaml_path, 'r') as file:
        labels_dic = yaml.safe_load(file) # Cargar diccionario de etiquetas
    labels_dic = labels_dic['names'] # Extraer nombres de etiquetas

    return keypoints_map_pos, classes_names_dic, labels_dic # Retornar diccionarios

# Función para crear diccionario de información de colores para equipos
def create_colors_info(
    team1_name,
    team1_p_color,
    team1_gk_color,
    team2_name,
    team2_p_color,
    team2_gk_color
    ):

    team1_p_color_rgb = ImageColor.getcolor(team1_p_color, "RGB") # Convertir color de jugadores equipo 1 a RGB
    team1_gk_color_rgb = ImageColor.getcolor(team1_gk_color, "RGB") # Convertir color de portero equipo 1 a RGB
    team2_p_color_rgb = ImageColor.getcolor(team2_p_color, "RGB") # Convertir color de jugadores equipo 2 a RGB
    team2_gk_color_rgb = ImageColor.getcolor(team2_gk_color, "RGB") # Convertir color de portero equipo 2 a RGB

    # Crear diccionario de colores
    colors_dic = {
        team1_name: [team1_p_color_rgb, team1_gk_color_rgb], # Colores equipo 1
        team2_name: [team2_p_color_rgb, team2_gk_color_rgb] # Colores equipo 2
    }
    # Definir lista de colores para usar en predicción de equipo de jugadores detectados
    colors_list = colors_dic[team1_name] + colors_dic[team2_name]
    # Convirtiendo color_list a espacio L*a*b*
    color_list_lab = [skimage.color.rgb2lab([i/255 for i in c]) for c in colors_list]

    return colors_dic, color_list_lab # Retornar diccionario y lista Lab

# Función para generar un nombre de archivo único para videos de salida
def generate_file_name():

    os.makedirs('./outputs/', exist_ok=True) # Crear directorio outputs si no existe
    list_video_files = os.listdir('./outputs/') # Listar archivos en outputs
    idx = 0 # Inicializar índice
    
    # Bucle para encontrar nombre único
    while True:
        idx += 1 # Incrementar índice
        output_file_name = f'detect_{idx}' # Generar nombre
        if output_file_name + '.mp4' not in list_video_files: # Verificar si existe
            break
    return output_file_name
