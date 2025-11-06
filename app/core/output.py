import cv2

def write_processed_video(processed_output, annotated_frame, output_file_name, fps, save_processed_separately):
    """
    Escribe el frame del video procesado si está habilitado.
    Args:
        processed_output: VideoWriter para video procesado
        annotated_frame: Frame anotado a escribir
        output_file_name: Nombre base para archivo de salida
        fps: FPS del video
        save_processed_separately: Si guardar video procesado
    Returns:
        processed_output actualizado
    """
    if save_processed_separately:
        # Redimensionar frame si es demasiado grande para mejor compatibilidad
        max_width, max_height = 1280, 720
        if annotated_frame.shape[1] > max_width or annotated_frame.shape[0] > max_height:
            scale = min(max_width / annotated_frame.shape[1], max_height / annotated_frame.shape[0])
            new_width = int(annotated_frame.shape[1] * scale)
            new_height = int(annotated_frame.shape[0] * scale)
            annotated_frame_resized = cv2.resize(annotated_frame, (new_width, new_height))
        else:
            annotated_frame_resized = annotated_frame

        if processed_output is None: # Inicializar VideoWriter en el primer frame
            height, width, _ = annotated_frame_resized.shape
            processed_output = cv2.VideoWriter(f'./outputs/{output_file_name}_processed.mp4',
                                               cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))
        processed_output.write(annotated_frame_resized)

    return processed_output

def write_tactical_video(tactical_output, tac_map_copy, output_file_name, fps, save_tactical_separately):
    """
    Escribe el frame del video táctico si está habilitado.
    Args:
        tactical_output: VideoWriter para video táctico
        tac_map_copy: Mapa táctico a escribir
        output_file_name: Nombre base para archivo de salida
        fps: FPS del video
        save_tactical_separately: Si guardar video táctico
    Returns:
        tactical_output actualizado
    """
    if save_tactical_separately:
        if tactical_output is None: # Inicializar VideoWriter en el primer frame
            tac_height, tac_width, _ = tac_map_copy.shape
            tactical_output = cv2.VideoWriter(f'./outputs/{output_file_name}_tactical.mp4',
                                              cv2.VideoWriter_fourcc(*'avc1'), fps, (tac_width, tac_height))
        tactical_output.write(tac_map_copy)

    return tactical_output

def write_combined_video(combined_output, final_img, output_file_name, fps, save_combined):
    """
    Escribe el frame del video combinado si está habilitado.
    Args:
        combined_output: VideoWriter para video combinado
        final_img: Imagen combinada a escribir
        output_file_name: Nombre base para archivo de salida
        fps: FPS del video
        save_combined: Si guardar video combinado
    Returns:
        combined_output actualizado
    """
    if save_combined:
        if combined_output is None: # Inicializar VideoWriter en el primer frame
            comb_height, comb_width, _ = final_img.shape
            combined_output = cv2.VideoWriter(f'./outputs/{output_file_name}_combined.mp4',
                                              cv2.VideoWriter_fourcc(*'avc1'), fps, (comb_width, comb_height))
        combined_output.write(final_img)

    return combined_output

def release_video_writers(processed_output, tactical_output, combined_output,
                         save_processed_separately, save_tactical_separately, save_combined, output_file_name):
    """
    Libera los escritores de video si existen.
    Args:
        processed_output: VideoWriter de video procesado
        tactical_output: VideoWriter de video táctico
        combined_output: VideoWriter de video combinado
        save_processed_separately: Si se guardó video procesado
        save_tactical_separately: Si se guardó video táctico
        save_combined: Si se guardó video combinado
        output_file_name: Nombre base para archivos de salida
    Returns:
        Tupla de nombres de archivos de salida
    """
    if save_processed_separately and processed_output is not None:
        processed_output.release()

    if save_tactical_separately and tactical_output is not None:
        tactical_output.release()

    if save_combined and combined_output is not None:
        combined_output.release()

    processed_name = f'{output_file_name}_processed.mp4' if save_processed_separately else None
    tactical_name = f'{output_file_name}_tactical.mp4' if save_tactical_separately else None
    combined_name = f'{output_file_name}_combined.mp4' if save_combined else None

    return processed_name, tactical_name, combined_name
