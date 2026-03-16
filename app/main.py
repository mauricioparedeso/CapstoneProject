"""
Knowledge Base Curator — API principal.

Arranca con:
    uvicorn app.main:app --reload
"""
from fastapi import FastAPI

from app.models.document import create_tables
from app.routers.documents import router as documents_router

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
