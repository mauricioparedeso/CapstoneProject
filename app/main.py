"""
Knowledge Base Curator — API principal.

Arranca con:
    pip install -r requirements.txt
    python -m uvicorn app.main:app --reload --reload-exclude "app/chroma_data/*"
"""
import re
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from app.models.document import create_tables
from app.routers.documents import router as documents_router

from app.Lang_Imp import APP as agente_grafo
from app.Lang_Imp import CONFIG
from pydantic import BaseModel

# ── Guardrails ────────────────────────────────────────────────────────────────

PROMPT_INJECTION_PATTERNS = [
    r"ignora\s+(tus\s+)?(instrucciones|reglas)",
    r"olvida\s+(tus\s+)?(instrucciones|reglas)",
    r"nuevo\s+(comportamiento|rol|sistema|prompt)",
    r"eres\s+ahora\s+un",
    r"actúa\s+como\s+si",
    r"\[system\]",
    r"\[instrucción\]",
    r"jailbreak",
    r"modo\s+sin\s+restricciones",
    r"bypass",
    r"ignore\s+(your\s+)?(instructions|rules)",
    r"you\s+are\s+now",
    r"disregard",
]

OUT_OF_SCOPE_PATTERNS = [
    r"^(dame\s+)?(una\s+)?receta",
    r"^(escríbeme\s+)?(un\s+)?poema",
    r"^(cuánto\s+es\s+)?\d+\s*[\+\-\*\/]\s*\d+",
    r"^(qué\s+es\s+)?(la\s+capital\s+de)",
    r"(hackear|hack|exploit|vulnerabilidad\s+de\s+seguridad)",
    r"(contraseña|password|credencial)",
]

def check_guardrails(message: str) -> tuple[bool, str]:
    if len(message.strip()) < 3:
        return False, "La pregunta es demasiado corta."
    if len(message) > 2000:
        return False, "La pregunta excede el límite de 2000 caracteres."
    msg_lower = message.lower()
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, msg_lower):
            return False, "La consulta contiene patrones no permitidos."
    for pattern in OUT_OF_SCOPE_PATTERNS:
        if re.search(pattern, msg_lower):
            return False, "Esta consulta está fuera del alcance del sistema. El agente está diseñado para analizar documentos curriculares."
    return True, ""

app = FastAPI(
    title="Knowledge Base Curator API",
    description=(
        "API para ingesta y gestión de documentos de la Knowledge Base. "
        "Parte del sistema de análisis curricular con IA."
    ),
    version="0.1.0",
)

create_tables()
app.include_router(documents_router)


@app.get("/health", tags=["health"])
def health_check():
    return {"status": "ok", "version": "0.1.0"}


# ── Endpoint de LangGraph ─────────────────────────────────────────────────────

class ChatQuery(BaseModel):
    message: str


@app.post("/agente/chat", tags=["agente"])
async def chatear_con_agente(query: ChatQuery):
    es_valido, error_msg = check_guardrails(query.message)
    if not es_valido:
        return {"error": error_msg, "pregunta": query.message, "respuesta": error_msg}

    try:
        inputs = {"messages": [query.message]}
        resultado = agente_grafo.invoke(inputs, config=CONFIG)
        ultimo_mensaje = resultado["messages"][-1]
        if isinstance(ultimo_mensaje, str):
            respuesta_final = ultimo_mensaje
        else:
            respuesta_final = ultimo_mensaje.content
        return {
            "pregunta": query.message,
            "respuesta": respuesta_final,
            "num_mensajes": len(resultado["messages"])
        }
    except Exception as e:
        import traceback
        return {
            "error": f"Error al ejecutar el agente: {str(e)}",
            "traceback": traceback.format_exc()
        }


# ── Endpoint de streaming ─────────────────────────────────────────────────────

NODOS_ORDEN = [
    "prompt_node",
    "memory_node",
    "planner_node",
    "tool_executor_node",
    "query_node",
    "writer_node",
]


@app.post("/agente/chat/stream", tags=["agente"])
async def chatear_con_agente_stream(query: ChatQuery):
    es_valido, error_msg = check_guardrails(query.message)
    if not es_valido:
        def error_stream():
            yield f"data: {json.dumps({'tipo': 'error', 'mensaje': error_msg})}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    def generar_eventos():
        try:
            inputs = {"messages": [query.message]}
            nodos_vistos = []
            total = len(NODOS_ORDEN)
            chunk = None

            for chunk in agente_grafo.stream(inputs, config={"recursion_limit": 50}):
                for nodo in chunk.keys():
                    if nodo in NODOS_ORDEN:
                        # tool_executor_node puede ejecutarse múltiples veces
                        if nodo != "tool_executor_node" and nodo in nodos_vistos:
                            continue
                        nodos_vistos.append(nodo)
                        pct = min(int((len(nodos_vistos) / total) * 100), 99)
                        evento = json.dumps({"tipo": "progreso", "nodo": nodo, "porcentaje": pct})
                        yield f"data: {evento}\n\n"

            # Forzar 100% antes de emitir la respuesta final
            yield f"data: {json.dumps({'tipo': 'progreso', 'nodo': 'writer_node', 'porcentaje': 100})}\n\n"

            # Extraer respuesta del último chunk
            respuesta = "Sin respuesta."
            if chunk:
                ultimo_nodo = list(chunk.keys())[-1]
                mensajes = chunk[ultimo_nodo].get("messages", [])
                if mensajes:
                    ultimo_msg = mensajes[-1]
                    respuesta = ultimo_msg.content if hasattr(ultimo_msg, "content") else str(ultimo_msg)

            fin = json.dumps({"tipo": "fin", "respuesta": respuesta})
            yield f"data: {fin}\n\n"

        except Exception as e:
            error = json.dumps({"tipo": "error", "mensaje": str(e)})
            yield f"data: {error}\n\n"

    return StreamingResponse(generar_eventos(), media_type="text/event-stream")


# ── Endpoint de búsqueda en ChromaDB ─────────────────────────────────────────

class SearchQuery(BaseModel):
    query: str
    n_results: int = 3


@app.post("/documents/search", tags=["documents"])
async def search_documents(body: SearchQuery):
    from app.Chroma_Imp import vector_store
    results = vector_store.similarity_search(body.query, k=body.n_results)
    return {
        "results": [
            {"source": r.metadata.get("source", "desconocido"), "content": r.page_content}
            for r in results
        ]
    }