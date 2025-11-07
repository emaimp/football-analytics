import streamlit as st
from ultralytics import YOLO

# Iniciar los roles
if "role" not in st.session_state:
    st.session_state.role = None

ROLE = ["admin"]
ROLE_PASSWORD = st.secrets["pass"]

#
# Hacer que el set_page_config no se ejecute
#
if "page_config" not in st.session_state:
    st.set_page_config(
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.session_state.page_config = True

#
# Pagina login de roles
#
def login():
    # Menu
    login_1, login_2, login_3 = st.columns(3)
    with login_2:
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("") # Espacio
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        
        # Título de la empresa
        title_1, title_2, title_3 = st.columns([5, 91, 4])
        with title_2:
            st.write("") # Espacio
            st.markdown('<h4 style="text-align:center; font-size: 2em;">⚽ Fútbol Computer Vision</h4>', unsafe_allow_html=True)
            st.write("") # Espacio
        
        # Formulario para manejar el botón enviar
        with st.form(key="login_form"):
            # Selección del rol
            role = st.selectbox("Elige un usuario", ROLE)
            
            st.write("") # Espacio
            
            # Entrada para la contraseña
            password = st.text_input(
                "Ingresa la contraseña", type="password", max_chars=10
            )
            st.write("") # Espacio
            
            # Botón de ingreso
            submit_button = st.form_submit_button("Entrar", type="primary", width="stretch")
            
            # Verificar si la contraseña no esta vacía
            if password != "":
                # Verificamos si el botón fue presionado o el formulario se envió
                if submit_button:
                    # Verificar la contraseña
                    if password == ROLE_PASSWORD.get(role, None):
                        st.session_state.role = role
                        st.success(f"Acceso concedido.")
                        st.rerun() # Reinicia la aplicación para reflejar el acceso
                    else:
                        st.error("Contraseña incorrecta.")
        
        badge_1, badge_2, badge_3 = st.columns([40, 30, 30])
        with badge_2:
            """
            [![GitHub](
                https://img.shields.io/badge/GitHub-black?style=for-the-badge&logo=github
                )](https://github.com/emaimp)
            """

# Pagina de logout
def logout():
    st.session_state.role = None
    st.rerun()
role = st.session_state.role

#
# Paginas
#
pages = {
    "Home": [
        st.Page(logout, title="Salir", icon=":material/logout:"),
        st.Page("views/💻_Pagina_Inicio.py", title="Inicio", default=(role == "admin")),
    ],
    "Archivos": [
        st.Page("views/💾_Uploader_File.py", title="Carga de Video"),
        st.Page("views/📽️_Reproductor.py", title="Reproductor"),
    ],
    "Configuración": [
        st.Page("views/🎨_Colores.py", title="Colores"),
        st.Page("views/⚙️_Parámetros.py", title="Parámetros"),
    ],
}

# Condicionales para la navegación
if st.session_state.role == "admin":

    # Cargar modelos si no están en session_state
    if "model_players" not in st.session_state:
        players_path = "app/models/Yolo8L_Players/best.pt"
        keypoints_path = "app/models/Yolo8M_Keypoints/best.pt"
        st.session_state.model_players = YOLO(players_path)
        st.session_state.model_keypoints = YOLO(keypoints_path)

    pg = st.navigation(pages)
else:
    pg = st.navigation([st.Page(login)])

pg.run() # Inicia la aplicación
