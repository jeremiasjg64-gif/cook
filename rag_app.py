import streamlit as st
import os
import shutil
import re
from pathlib import Path
import pyrebase

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from sentence_transformers import CrossEncoder
import numpy as np

# ============================================
# CONFIGURACIÓN DE FIREBASE (NUEVA APP)
# ============================================
# Reemplaza estos valores con los de tu nueva app de Firebase (proyecto RAG)
FIREBASE_CONFIG = {
    "apiKey": "AIzaSyAkx-rjtOODGq90ylMRCjTS_2nqnelrEbQ",                # Ejemplo: "AIzaSy..."
    "authDomain": "cook-a7ba8.firebaseapp.com",
    "databaseURL": "https://cook-a7ba8-default-rtdb.firebaseio.com",  # Opcional pero requerido por pyrebase
    "projectId": "cook-a7ba8",
    "storageBucket": "cook-a7ba8.firebasestorage.app",
    "messagingSenderId": "573777071011",
    "appId": "1:573777071011:web:a505d0373a25ae3f13a816"
}

# Inicializar Firebase Auth
firebase = pyrebase.initialize_app(FIREBASE_CONFIG)
auth = firebase.auth()

# ============================================
# CONFIGURACIÓN DEL RAG
# ============================================
BASE_DIR = Path("./documentos")
INDEX_BASE = Path("./faiss_index")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

st.set_page_config(page_title="RAG Inteligente - Soporte ENLACES", page_icon="📂", layout="wide")

# ============================================
# FUNCIONES DE AUTENTICACIÓN
# ============================================
def validar_email(email):
    patron = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(patron, email) is not None

def mostrar_login():
    st.markdown("""
    <style>
        .main .block-container {
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 2rem;
        }
        .login-card {
            background: white;
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            max-width: 400px;
            width: 100%;
        }
        .stTextInput > div > div > input {
            border-radius: 30px !important;
        }
        .stButton > button {
            border-radius: 30px !important;
            background-color: #6b4f3c !important;
            color: white !important;
        }
        .stButton > button:hover {
            background-color: #4e3a2b !important;
        }
    </style>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.markdown("<h1 style='text-align:center;'>📚 Asistente ENLACES</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align:center; color:#6b4f3c;'>🔐 Acceso al sistema</h3>", unsafe_allow_html=True)

        modo = st.radio("Selecciona acción:", ["Iniciar sesión", "Registrarse"], horizontal=True)

        with st.form("login_form"):
            email = st.text_input("📧 Correo electrónico", placeholder="usuario@ejemplo.com")
            password = st.text_input("🔑 Contraseña", type="password", placeholder="mínimo 6 caracteres")
            submit = st.form_submit_button("🚪 Continuar", use_container_width=True)

            if submit:
                if not email or not password:
                    st.error("❌ Completa ambos campos.")
                elif not validar_email(email):
                    st.error("❌ Formato de correo inválido.")
                else:
                    try:
                        if modo == "Iniciar sesión":
                            auth.sign_in_with_email_and_password(email, password)
                            st.session_state.autenticado = True
                            st.session_state.user_email = email
                            st.rerun()
                        else:
                            auth.create_user_with_email_and_password(email, password)
                            st.success("✅ Registro exitoso. Ahora inicia sesión.")
                    except Exception as e:
                        error = str(e)
                        if "EMAIL_NOT_FOUND" in error:
                            st.error("❌ Correo no registrado.")
                        elif "INVALID_PASSWORD" in error:
                            st.error("❌ Contraseña incorrecta.")
                        elif "EMAIL_EXISTS" in error:
                            st.error("❌ El correo ya está registrado.")
                        elif "WEAK_PASSWORD" in error:
                            st.error("❌ Contraseña débil (mínimo 6 caracteres).")
                        else:
                            st.error(f"❌ Error: {error}")
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# FUNCIONES DEL RAG (iguales a las originales)
# ============================================
@st.cache_resource
def cargar_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

@st.cache_resource
def cargar_cross_encoder():
    return CrossEncoder(CROSS_ENCODER_MODEL)

@st.cache_resource
def cargar_index(index_path):
    if os.path.exists(index_path):
        embeddings = cargar_embeddings()
        try:
            return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            st.error(f"Error al cargar índice: {e}")
    return None

def extraer_bloques_problema_solucion(texto):
    lineas = texto.split('\n')
    bloques = []
    bloque_actual = {"problema": "", "solucion": "", "texto_completo": ""}
    capturando = None
    for linea in lineas:
        linea_strip = linea.strip()
        if linea_strip.startswith("Problema:"):
            if capturando == "solucion":
                if bloque_actual["problema"] and bloque_actual["solucion"]:
                    bloques.append(bloque_actual.copy())
                bloque_actual = {"problema": "", "solucion": "", "texto_completo": ""}
            capturando = "problema"
            bloque_actual["problema"] = linea_strip
            bloque_actual["texto_completo"] += linea + "\n"
        elif linea_strip.startswith("Solución:"):
            capturando = "solucion"
            bloque_actual["solucion"] = linea_strip
            bloque_actual["texto_completo"] += linea + "\n"
        else:
            if capturando == "problema":
                bloque_actual["problema"] += " " + linea_strip
                bloque_actual["texto_completo"] += linea + "\n"
            elif capturando == "solucion":
                bloque_actual["solucion"] += " " + linea_strip
                bloque_actual["texto_completo"] += linea + "\n"
    if bloque_actual["problema"] and bloque_actual["solucion"]:
        bloques.append(bloque_actual)
    return bloques

def cargar_documentos_seleccionados(archivos):
    docs_originales = []
    for archivo in archivos:
        ruta = BASE_DIR / archivo
        if ruta.exists():
            try:
                loader = TextLoader(str(ruta), encoding="utf-8")
                docs = loader.load()
                for doc in docs:
                    bloques = extraer_bloques_problema_solucion(doc.page_content)
                    for bloque in bloques:
                        doc_bloque = Document(
                            page_content=bloque["texto_completo"],
                            metadata={
                                "source": archivo,
                                "problema": bloque["problema"],
                                "solucion": bloque["solucion"]
                            }
                        )
                        docs_originales.append(doc_bloque)
            except Exception as e:
                st.error(f"Error al cargar {archivo}: {e}")
        else:
            st.error(f"Archivo no encontrado: {archivo}")
    return docs_originales

def construir_index(archivos, index_path):
    if not archivos:
        st.error("No se seleccionó ningún archivo.")
        return False

    with st.spinner("Cargando y procesando documentos..."):
        documentos = cargar_documentos_seleccionados(archivos)
    if not documentos:
        st.error("No se pudieron cargar bloques válidos.")
        return False

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documentos)

    with st.spinner(f"Generando embeddings para {len(chunks)} fragmentos..."):
        embeddings = cargar_embeddings()
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(index_path)
    st.success(f"Índice guardado en {index_path} con {len(chunks)} fragmentos.")
    return True

def buscar_respuesta(pregunta, vector_store, cross_encoder, k=10):
    docs = vector_store.similarity_search(pregunta, k=k)
    if not docs:
        return None
    pairs = [(pregunta, doc.page_content) for doc in docs]
    scores = cross_encoder.predict(pairs)
    best_idx = np.argmax(scores)
    best_doc = docs[best_idx]
    if "solucion" in best_doc.metadata:
        return best_doc.metadata["solucion"]
    else:
        match = re.search(r'Solución:(.*?)(?:\n\n|\Z)', best_doc.page_content, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return best_doc.page_content[:500]

# ============================================
# PROGRAMA PRINCIPAL
# ============================================
if "autenticado" not in st.session_state:
    st.session_state.autenticado = False
    st.session_state.user_email = ""

if not st.session_state.autenticado:
    mostrar_login()
    st.stop()

# A partir de aquí el usuario está autenticado
# Mostrar barra lateral con datos del usuario y logout
with st.sidebar:
    st.markdown(f"**👤 Usuario:** {st.session_state.user_email}")
    if st.button("🚪 Cerrar sesión"):
        st.session_state.autenticado = False
        st.session_state.user_email = ""
        st.rerun()
    st.markdown("---")

# Resto de la interfaz RAG (igual que antes)
st.title("📚 Asistente Inteligente ENLACES")
st.markdown("Pregunta sobre cuentas, garantías, soporte técnico.")

if "mensajes" not in st.session_state:
    st.session_state.mensajes = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "cross_encoder" not in st.session_state:
    with st.spinner("Cargando modelo de re-ranking (solo la primera vez)..."):
        st.session_state.cross_encoder = cargar_cross_encoder()

with st.sidebar:
    st.header("📁 Selección de documentos")
    opcion = st.radio(
        "¿Qué documento(s) quieres usar?",
        ["📄 Solo cuentas_clases.txt", "📄 Solo soporte_equipos.txt", "📚 Ambos"]
    )
    if opcion == "📄 Solo cuentas_clases.txt":
        archivos = ["cuentas_clases.txt"]
        index_name = "index_cuentas"
    elif opcion == "📄 Solo soporte_equipos.txt":
        archivos = ["soporte_equipos.txt"]
        index_name = "index_soporte"
    else:
        archivos = ["cuentas_clases.txt", "soporte_equipos.txt"]
        index_name = "index_ambos"

    INDEX_PATH = INDEX_BASE / index_name

    if st.button("🔄 Construir / Reconstruir índice", type="primary"):
        if construir_index(archivos, str(INDEX_PATH)):
            st.session_state.vector_store = cargar_index(str(INDEX_PATH))
            st.session_state.mensajes = []
            st.rerun()

    if st.button("🗑️ Limpiar índice"):
        if os.path.exists(INDEX_PATH):
            shutil.rmtree(INDEX_PATH)
        st.session_state.vector_store = None
        st.session_state.mensajes = []
        st.success("Índice eliminado.")
        st.rerun()

for msg in st.session_state.mensajes:
    with st.chat_message(msg["rol"]):
        st.markdown(msg["contenido"])
        if "fuentes" in msg and msg["fuentes"]:
            st.caption(f"📂 Fuente: {msg['fuentes']}")

if prompt := st.chat_input("Escribe tu pregunta:"):
    st.session_state.mensajes.append({"rol": "user", "contenido": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.vector_store is None:
        respuesta = "⚠️ No hay índice. Selecciona documentos y haz clic en 'Construir / Reconstruir índice'."
        with st.chat_message("assistant"):
            st.markdown(respuesta)
        st.session_state.mensajes.append({"rol": "assistant", "contenido": respuesta, "fuentes": ""})
    else:
        with st.chat_message("assistant"):
            with st.spinner("Buscando la solución más relevante..."):
                solucion = buscar_respuesta(prompt, st.session_state.vector_store, st.session_state.cross_encoder)
            if solucion:
                st.markdown(solucion)
                st.session_state.mensajes.append({"rol": "assistant", "contenido": solucion, "fuentes": "Documentos seleccionados"})
            else:
                st.markdown("No se encontró una solución relacionada. Intenta reformular la pregunta.")
                st.session_state.mensajes.append({"rol": "assistant", "contenido": "No se encontró una solución relacionada.", "fuentes": ""})