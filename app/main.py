"""
Knowledge Base Curator — API principal.

Arranca con:
    pip install -r requirements.txt
    python -m uvicorn app.main:app --reload --reload-exclude "app/chroma_data/*"
    
    python -m uvicorn app.main:app --reload --reload-exclude "app/chroma_data"
    https://reimagined-acorn-xjjq9gq564x2j6v-8000.app.github.dev/docs
"""
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
        # Invocamos al grafo con el mensaje del usuario (como string simple)
        inputs = {"messages": [query.message]}
        resultado = agente_grafo.invoke(inputs, config={"recursion_limit": 50})
        
        # Extraemos el contenido del último mensaje (la respuesta del asistente)
        ultimo_mensaje = resultado["messages"][-1]
        
        # Manejar ambos tipos de mensajes (string o objeto con .content)
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

