import cv2

# Función para escribir el frame del video procesado si está habilitado
def write_processed_video(
    processed_output,
    annotated_frame,
    output_file_name,
    fps,
    save_processed_separately
    ):

    # Si se debe guardar el video procesado por separado
    if save_processed_separately:
        # Redimensionar frame si es demasiado grande para mejor compatibilidad
        max_width, max_height = 1280, 720 # Ancho y alto máximos permitidos para compatibilidad
        
        # Verificar si el frame es demasiado grande
        if annotated_frame.shape[1] > max_width or annotated_frame.shape[0] > max_height:
            scale = min(max_width / annotated_frame.shape[1], max_height / annotated_frame.shape[0]) # Calcular la escala de redimensionamiento
            new_width = int(annotated_frame.shape[1] * scale) # Nuevo ancho calculado
            new_height = int(annotated_frame.shape[0] * scale) # Nuevo alto calculado
            annotated_frame_resized = cv2.resize(annotated_frame, (new_width, new_height)) # Redimensionar el frame anotado
        else:
            annotated_frame_resized = annotated_frame # Usar el frame original si no necesita redimensionar
        
        # Inicializar VideoWriter en el primer frame si no existe
        if processed_output is None:
            height, width, _ = annotated_frame_resized.shape # Obtener las dimensiones del frame redimensionado
            processed_output = cv2.VideoWriter(f'./outputs/{output_file_name}_processed.mp4', # Crear VideoWriter para video procesado
                                               cv2.VideoWriter_fourcc(*'avc1'), # Codec de video
                                               fps, (width, height)) # FPS y dimensiones
        processed_output.write(annotated_frame_resized) # Escribir el frame redimensionado en el video

    return processed_output # Retornar el VideoWriter actualizado

# Función para escribir el frame del video táctico si está habilitado
def write_tactical_video(
    tactical_output,
    tac_map_copy,
    output_file_name,
    fps,
    save_tactical_separately
    ):

    # Si se debe guardar el video táctico por separado
    if save_tactical_separately:
        
        # Inicializar VideoWriter en el primer frame si no existe
        if tactical_output is None:
            tac_height, tac_width, _ = tac_map_copy.shape # Obtener las dimensiones del mapa táctico
            tactical_output = cv2.VideoWriter(f'./outputs/{output_file_name}_tactical.mp4', # Crear VideoWriter para video táctico
                                              cv2.VideoWriter_fourcc(*'avc1'), # Codec de video
                                              fps, (tac_width, tac_height)) # FPS y dimensiones
        tactical_output.write(tac_map_copy) # Escribir el mapa táctico en el video

    return tactical_output # Retornar el VideoWriter actualizado

# Función para escribir el frame del video combinado si está habilitado
def write_combined_video(
    combined_output,
    final_img,
    output_file_name,
    fps,
    save_combined
    ):

    # Si se debe guardar el video combinado
    if save_combined:
        
        # Inicializar VideoWriter en el primer frame si no existe
        if combined_output is None:
            comb_height, comb_width, _ = final_img.shape # Obtener las dimensiones de la imagen combinada
            combined_output = cv2.VideoWriter(f'./outputs/{output_file_name}_combined.mp4', # Crear VideoWriter para video combinado
                                              cv2.VideoWriter_fourcc(*'avc1'), # Codec de video
                                              fps, (comb_width, comb_height)) # FPS y dimensiones
        combined_output.write(final_img) # Escribir la imagen combinada en el video

    return combined_output # Retornar el VideoWriter actualizado

# Función para liberar los escritores de video si existen
def release_video_writers(
    processed_output,
    tactical_output,
    combined_output,
    save_processed_separately,
    save_tactical_separately,
    save_combined,
    output_file_name
    ):

    # Liberar el VideoWriter de video procesado si existe
    if save_processed_separately and processed_output is not None:
        processed_output.release()

    # Liberar el VideoWriter de video táctico si existe
    if save_tactical_separately and tactical_output is not None:
        tactical_output.release()

    # Liberar el VideoWriter de video combinado si existe
    if save_combined and combined_output is not None:
        combined_output.release()

    # Nombre del archivo procesado si se guardó, sino None
    processed_name = f'{output_file_name}_processed.mp4' if save_processed_separately else None
    # Nombre del archivo táctico si se guardó, sino None
    tactical_name = f'{output_file_name}_tactical.mp4' if save_tactical_separately else None
    # Nombre del archivo combinado si se guardó, sino None
    combined_name = f'{output_file_name}_combined.mp4' if save_combined else None

    return processed_name, tactical_name, combined_name # Retornar tupla con los nombres de los archivos de salida
