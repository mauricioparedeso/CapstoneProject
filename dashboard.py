"""
E4 — Instructor Dashboard
Knowledge Base Curator — SoftServe

Correr con:
    streamlit run dashboard.py

Requiere que la API esté corriendo en:
    https://literate-capybara-g456xxg95x652979q-8000.app.github.dev
"""

import streamlit as st
import requests
import json
from datetime import datetime

# ── Configuración ─────────────────────────────────────────────────────────────

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Knowledge Base Curator",
    page_icon="KB",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colores SoftServe ─────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Paleta SoftServe */
    :root {
        --ss-blue: #0099D8;
        --ss-blue-light: #5BC4F0;
        --ss-dark: #2C2C2C;
        --ss-gray: #6C6C6C;
        --ss-light: #DCE3E7;
        --ss-white: #FFFFFF;
    }

    /* Header principal */
    .ss-header {
        background-color: var(--ss-blue);
        color: white;
        padding: 1.2rem 2rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .ss-header h1 {
        margin: 0;
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .ss-header p {
        margin: 0;
        font-size: 0.9rem;
        opacity: 0.9;
    }

    /* Tarjetas de documentos */
    .doc-card {
        background: white;
        border: 1px solid var(--ss-light);
        border-left: 4px solid var(--ss-blue);
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        transition: box-shadow 0.2s;
    }
    .doc-card:hover {
        box-shadow: 0 2px 8px rgba(0,153,216,0.15);
    }
    .doc-title {
        font-weight: 600;
        color: var(--ss-dark);
        font-size: 0.95rem;
        margin-bottom: 0.3rem;
    }
    .doc-meta {
        color: var(--ss-gray);
        font-size: 0.8rem;
    }
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    .badge-pdf { background: #FFE8E2; color: #C13A1A; }
    .badge-docx { background: #E2EEFF; color: #1A4FC1; }
    .badge-txt { background: #E8F5E9; color: #1B7B2E; }

    /* Chat */
    .chat-user {
        color: #1a1a1a !important;
        background: #FFF3F0;
        border-left: 3px solid var(--ss-blue);
        border-radius: 4px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .chat-agent {
        color: #1a1a1a !important;
        background: #F5F5F5;
        border-left: 3px solid var(--ss-gray);
        border-radius: 4px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .chat-user * { color: #1a1a1a !important; }
    .chat-agent * { color: #1a1a1a !important; }
    .chat-label {
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .chat-label.user { color: var(--ss-blue); }
    .chat-label.agent { color: var(--ss-gray); }

    /* Sugerencias placeholder */
    .suggestion-card {
        background: white;
        border: 1px solid var(--ss-light);
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        opacity: 0.7;
    }
    .coming-soon {
        background: #FFF8F7;
        border: 2px dashed var(--ss-blue-light);
        border-radius: 8px;
        padding: 2rem;
        text-align: center;
        color: var(--ss-gray);
    }

    /* Sidebar — fondo oscuro y todo el texto en blanco */
    [data-testid="stSidebar"] {
        background-color: #2C2C2C !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] strong {
        color: white !important;
    }

    /* Botones */
    .stButton > button {
        background-color: var(--ss-blue);
        color: white;
        border: none;
        border-radius: 4px;
        font-weight: 500;
    }
    .stButton > button:hover {
        background-color: #007DAF;
        color: white;
    }

    /* Métricas */
    [data-testid="metric-container"] {
        background: white;
        border: 1px solid var(--ss-light);
        border-radius: 8px;
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers API ───────────────────────────────────────────────────────────────

def get_documents():
    try:
        r = requests.get(f"{API_BASE}/documents/", timeout=10)
        if r.status_code == 200:
            return r.json().get("documents", [])
    except Exception:
        pass
    return None


def upload_document(file):
    try:
        r = requests.post(
            f"{API_BASE}/documents/upload",
            files={"file": (file.name, file.read(), file.type)},
            timeout=30,
        )
        return r.json(), r.status_code
    except Exception as e:
        return {"error": str(e)}, 500


def chat_with_agent(message):
    try:
        r = requests.post(
            f"{API_BASE}/agente/chat",
            json={"message": message},
            timeout=60,
        )
        if r.status_code == 200:
            return r.json().get("respuesta", "Sin respuesta del agente.")
    except Exception as e:
        return f"Error al conectar con el agente: {e}"
    return "Error inesperado."


def chat_with_agent_stream(message, placeholder_progreso):
    """
    Llama al endpoint de streaming y actualiza un placeholder con el progreso.
    Retorna la respuesta final del agente.
    """
    NODOS_ICONO = {
        "memory_node":        "Consultando memoria",
        "prompt_node":        "Analizando la pregunta",
        "planner_node":       "Planificando tareas",
        "tool_executor_node": "Ejecutando herramientas",
        "query_node":         "Generando sugerencias",
        "writer_node":        "Redactando respuesta final",
    }
    NODOS_ORDEN = list(NODOS_ICONO.keys())
    TOTAL = len(NODOS_ORDEN)

    respuesta_final = ""
    pasos_completados = []

    try:
        r = requests.post(
            f"{API_BASE}/agente/chat/stream",
            json={"message": message},
            timeout=120,
        )

        eventos = r.text.strip().split("\n\n")
        for evento in eventos:
            for line in evento.split("\n"):
                if not line.startswith("data:"):
                    continue
                try:
                    data = json.loads(line[5:].strip())
                except Exception:
                    continue

                if data.get("tipo") == "progreso":
                    nodo = data.get("nodo", "")
                    label = NODOS_ICONO.get(nodo, nodo)
                    pct = data.get("porcentaje", 0)
                    pasos_completados.append((label, pct))

                    with placeholder_progreso.container():
                        st.markdown(f"""
                        <div style="background:#1a1a2e; border:1px solid #0099D8; border-radius:10px; padding:1rem 1.5rem; margin:0.5rem 0;">
                            <p style="color:#0099D8; font-weight:700; font-size:0.85rem; margin:0 0 0.8rem 0; letter-spacing:1px;">PROCESANDO PIPELINE</p>
                        """, unsafe_allow_html=True)
                        st.progress(pct, text=f"{label}... {pct}%")
                        for paso_label, paso_pct in pasos_completados[:-1]:
                            st.markdown(f"<p style='color:#aaa; font-size:0.82rem; margin:0.2rem 0;'>✓ {paso_label}</p>", unsafe_allow_html=True)
                        st.markdown(f"<p style='color:#0099D8; font-size:0.85rem; font-weight:600; margin:0.2rem 0;'>▶ {label}</p>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                elif data.get("tipo") == "fin":
                    respuesta_final = data.get("respuesta", "")
                    with placeholder_progreso.container():
                        st.markdown(f"""
                        <div style="background:#0d2b1a; border:1px solid #00c853; border-radius:10px; padding:1rem 1.5rem; margin:0.5rem 0;">
                            <p style="color:#00c853; font-weight:700; font-size:0.85rem; margin:0 0 0.5rem 0; letter-spacing:1px;">PIPELINE COMPLETADO</p>
                        """, unsafe_allow_html=True)
                        st.progress(100, text="Completado — 100%")
                        for paso_label, _ in pasos_completados:
                            st.markdown(f"<p style='color:#aaa; font-size:0.82rem; margin:0.2rem 0;'>✓ {paso_label}</p>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                elif data.get("tipo") == "error":
                    respuesta_final = f"Error: {data.get('mensaje', 'desconocido')}"

    except Exception as e:
        respuesta_final = f"Error al conectar con el agente: {e}"

    return respuesta_final


def check_api():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="ss-header">
    <div>
        <h1>Knowledge Base Curator</h1>
        <p>Panel de gestión curricular con IA · SoftServe</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Estado de la API
api_ok = check_api()
if api_ok:
    st.success("API conectada y funcionando")
else:
    st.error("No se puede conectar con la API. Verifica que este corriendo.")


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Navegación")
    pagina = st.radio(
        "",
        ["Documentos", "Consultar Agente", "Sugerencias"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**API**")
    st.markdown(f"<p style='color:#0099D8; font-size:0.75rem; word-break:break-all;'>{API_BASE}</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Épica E4** — Instructor Dashboard")
    st.markdown("Sprint 2 · En progreso")


# ── Página: Documentos ────────────────────────────────────────────────────────

if pagina == "Documentos":
    st.subheader("Knowledge Base — Documentos")

    # Upload
    with st.expander("Subir nuevo documento", expanded=False):
        archivo = st.file_uploader(
            "Selecciona un archivo",
            type=["pdf", "docx", "txt"],
            help="Formatos soportados: PDF, DOCX, TXT",
        )
        if archivo and st.button("Subir documento"):
            with st.spinner("Subiendo..."):
                resultado, status = upload_document(archivo)
            if status == 201:
                st.success(f"Documento subido. ID: `{resultado['document']['id']}`")
                st.rerun()
            else:
                st.error(f"Error: {resultado.get('detail', resultado)}")

    st.markdown("---")

    # Lista de documentos
    docs = get_documents()

    if docs is None:
        st.error("No se pudo obtener la lista de documentos.")
    elif len(docs) == 0:
        st.info("No hay documentos en la Knowledge Base. Sube el primero.")
    else:
        # Métricas
        col1, col2, col3 = st.columns(3)
        formatos = [d["file_format"] for d in docs]
        col1.metric("Total documentos", len(docs))
        col2.metric("PDFs", formatos.count("pdf"))
        col3.metric("Otros formatos", formatos.count("docx") + formatos.count("txt"))

        st.markdown("---")

        # Filtros
        col_f1, col_f2 = st.columns([2, 1])
        buscar = col_f1.text_input("Buscar documento", placeholder="Nombre del archivo...")
        formato_filtro = col_f2.selectbox("Formato", ["Todos", "pdf", "docx", "txt"])

        docs_filtrados = docs
        if buscar:
            docs_filtrados = [d for d in docs_filtrados if buscar.lower() in d["original_filename"].lower()]
        if formato_filtro != "Todos":
            docs_filtrados = [d for d in docs_filtrados if d["file_format"] == formato_filtro]

        st.markdown(f"**{len(docs_filtrados)} documento(s)**")

        for doc in docs_filtrados:
            fecha = doc["uploaded_at"][:10] if doc.get("uploaded_at") else "—"
            size_kb = round(doc["file_size_bytes"] / 1024, 1)
            fmt = doc["file_format"]
            badge_class = f"badge-{fmt}"

            st.markdown(f"""
            <div class="doc-card">
                <div class="doc-title">{doc["original_filename"]}</div>
                <div class="doc-meta">
                    <span class="badge {badge_class}">{fmt.upper()}</span>
                    &nbsp;·&nbsp; {size_kb} KB
                    &nbsp;·&nbsp; Subido el {fecha}
                    &nbsp;·&nbsp; <code style="font-size:0.75rem">{doc["id"][:8]}...</code>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ── Página: Consultar Agente ──────────────────────────────────────────────────

elif pagina == "Consultar Agente":
    st.subheader("Consultar Agente IA")
    st.caption("El agente puede responder preguntas sobre los documentos de la Knowledge Base.")

    # Historial de chat en session_state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Mostrar historial
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="chat-user">
                <div class="chat-label user">Instructor</div>
                {msg["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-agent">
                <div class="chat-label agent">Agente IA</div>
                {msg["content"]}
            </div>
            """, unsafe_allow_html=True)

    # Preguntas sugeridas
    if len(st.session_state.chat_history) == 0:
        st.markdown("**Preguntas sugeridas:**")
        col1, col2 = st.columns(2)
        if col1.button("¿Qué archivos hay en la Knowledge Base?"):
            st.session_state.chat_input = "¿Qué archivos hay en la Knowledge Base?"
        if col2.button("Resume el contenido de los documentos"):
            st.session_state.chat_input = "Resume el contenido de los documentos disponibles"

    # Input
    pregunta = st.text_input(
        "Escribe tu pregunta",
        value=st.session_state.get("chat_input", ""),
        placeholder="Ej: ¿Qué temas cubre el documento X?",
        key="chat_input_field",
    )

    col_send, col_clear = st.columns([1, 5])
    enviar = col_send.button("Enviar")
    if col_clear.button("Limpiar chat"):
        st.session_state.chat_history = []
        st.rerun()

    if enviar and pregunta.strip():
        st.session_state.chat_history.append({"role": "user", "content": pregunta})
        if "chat_input" in st.session_state:
            del st.session_state["chat_input"]
        # Crear placeholder para el progreso
        placeholder = st.empty()
        respuesta = chat_with_agent_stream(pregunta, placeholder)
        # Limpiar el placeholder de progreso
        placeholder.empty()
        # Guardar respuesta
        if respuesta and respuesta.strip():
            st.session_state.chat_history.append({"role": "agent", "content": respuesta})
        else:
            respuesta_fallback = chat_with_agent(pregunta)
            st.session_state.chat_history.append({"role": "agent", "content": respuesta_fallback})
        st.rerun()


# ── Página: Sugerencias ───────────────────────────────────────────────────────

elif pagina == "Sugerencias":
    st.subheader("Sugerencias del Agente IA")
    st.caption("Revisión y aprobación de propuestas generadas por el sistema.")

    st.markdown("""
    <div class="coming-soon">
        <h3 style="color:#0099D8; margin-bottom:0.5rem">Proximamente disponible</h3>
        <p style="margin:0">Esta sección estará activa cuando E3 genere sugerencias estructuradas.<br>
        El endpoint <code>POST /suggestions/review</code> está reservado para esta integración.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Estructura esperada de sugerencias (E3 → E4):**")

    ejemplo = {
        "id": "sugg-001",
        "document": "syllabus_2026.pdf",
        "type": "outdated_content",
        "proposal": "Actualizar la sección de redes neuronales con contenido de 2024.",
        "rationale": "El agente detectó que el capítulo 4 referencia frameworks obsoletos.",
        "sources": ["doc_A.pdf p.12", "doc_B.pdf p.3"],
        "status": "pending",
    }
    st.json(ejemplo)
    st.caption("Este JSON es el contrato de datos que E3 deberá entregar cuando esté implementado.")