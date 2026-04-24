"""
Knowledge Base Curator — API principal.

Arranca con:
    pip install -r requirements.txt
    python -m uvicorn app.main:app --reload --reload-exclude "app/chroma_data/*"
    https://reimagined-acorn-xjjq9gq564x2j6v-8000.app.github.dev/docs
"""
from fastapi.responses import StreamingResponse
import json
import asyncio

from fastapi import FastAPI

from app.models.document import create_tables
from app.routers.documents import router as documents_router

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.Lang_Imp import APP as agente_grafo  # Importamos el grafo compilado
from pydantic import BaseModel

app = FastAPI(
    title="Knowledge Base Curator API",
    description=(
        "API para ingesta y gestión de documentos de la Knowledge Base. "
        "Parte del sistema de análisis curricular con IA."
    ),
    version="0.1.0",
)

# Crear tablas al iniciar (en producción se usaría Alembic)
create_tables()

# Registrar routers
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
        # Invocamos al grafo con el mensaje del usuario
        inputs = {"messages": [("user", query.message)]}
        resultado = agente_grafo.invoke(inputs, config={"recursion_limit": 50} )
        
        # Extraemos el contenido del último mensaje (la respuesta del asistente)
        respuesta_final = resultado["messages"][-1].content
        
        return {
            "pregunta": query.message,
            "respuesta": respuesta_final,
            "num_mensajes": len(resultado["messages"])
        }
    except Exception as e:
        return {"error": f"Error al ejecutar el agente: {str(e)}"}

@app.post("/agente/chat/stream", tags=["agente"])
async def chatear_con_agente_stream(query: ChatQuery):
    """
    Ejecuta el grafo de LangGraph con streaming.
    Emite Server-Sent Events con el nodo actual y porcentaje de progreso.
    """
    NODOS_ORDEN = [
        "memory_node",
        "prompt_node",
        "planner_node",
        "tool_executor_node",
        "query_node",
        "writer_node",
    ]
    NODOS_LABEL = {
        "memory_node":        "Consultando memoria...",
        "prompt_node":        "Analizando la pregunta...",
        "planner_node":       "Planificando tareas...",
        "tool_executor_node": "Ejecutando herramientas...",
        "query_node":         "Generando sugerencias...",
        "writer_node":        "Redactando respuesta...",
    }

    async def generate():
        try:
            inputs = {"messages": [("user", query.message)]}
            nodos_vistos = []
            respuesta_final = ""

            for chunk in agente_grafo.stream(
                inputs,
                config={"recursion_limit": 50},
                stream_mode="updates",
            ):
                for node_name, node_output in chunk.items():
                    if node_name not in nodos_vistos:
                        nodos_vistos.append(node_name)

                    porcentaje = int(
                        (len(nodos_vistos) / len(NODOS_ORDEN)) * 100
                    )
                    porcentaje = min(porcentaje, 95)  # reservar 100% para el final

                    evento = {
                        "tipo": "progreso",
                        "nodo": node_name,
                        "label": NODOS_LABEL.get(node_name, node_name),
                        "porcentaje": porcentaje,
                        "nodos_completados": nodos_vistos,
                    }
                    yield f"data: {json.dumps(evento)}\n\n"
                    await asyncio.sleep(0)  # ceder control al event loop

                    # Capturar respuesta del writer
                    if node_name == "writer_node":
                        msgs = node_output.get("messages", [])
                        if msgs:
                            respuesta_final = msgs[-1].content

            # Evento final con la respuesta
            yield f"data: {json.dumps({'tipo': 'fin', 'porcentaje': 100, 'respuesta': respuesta_final})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'tipo': 'error', 'mensaje': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
