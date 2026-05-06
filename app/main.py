"""
Knowledge Base Curator — API principal.

Arranca con:
    pip install -r requirements.txt
    python -m uvicorn app.main:app --reload --reload-exclude "app/chroma_data/*"
    
    python -m uvicorn app.main:app --reload --reload-exclude "app/chroma_data"
    https://reimagined-acorn-xjjq9gq564x2j6v-8000.app.github.dev/docs
"""
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from app.models.document import create_tables
from app.routers.documents import router as documents_router

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.Lang_Imp import APP as agente_grafo
from pydantic import BaseModel

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
    """Endpoint de salud — confirma que la API está corriendo."""
    return {"status": "ok", "version": "0.1.0"}


#=================Endpoint de LangGraph=================
class ChatQuery(BaseModel):
    message: str


@app.post("/agente/chat", tags=["agente"])
async def chatear_con_agente(query: ChatQuery):
    """
    Ejecuta el grafo de LangGraph y devuelve la respuesta final del agente.
    """
    try:
        inputs = {"messages": [query.message]}
        resultado = agente_grafo.invoke(inputs, config={"recursion_limit": 50})
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


#=================Endpoint de streaming=================
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
    """
    Ejecuta el grafo de LangGraph emitiendo eventos SSE por cada nodo.
    """
    def generar_eventos():
        try:
            total = len(NODOS_ORDEN)

            for i, nodo in enumerate(NODOS_ORDEN):
                pct = int(((i + 1) / total) * 100)
                evento = json.dumps({"tipo": "progreso", "nodo": nodo, "porcentaje": pct})
                yield f"data: {evento}\n\n"

            inputs = {"messages": [query.message]}
            resultado = agente_grafo.invoke(inputs, config={"recursion_limit": 50})
            ultimo_mensaje = resultado["messages"][-1]
            respuesta = ultimo_mensaje.content if hasattr(ultimo_mensaje, "content") else str(ultimo_mensaje)

            fin = json.dumps({"tipo": "fin", "respuesta": respuesta})
            yield f"data: {fin}\n\n"

        except Exception as e:
            error = json.dumps({"tipo": "error", "mensaje": str(e)})
            yield f"data: {error}\n\n"

    return StreamingResponse(generar_eventos(), media_type="text/event-stream")