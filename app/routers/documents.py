"""
Router de documentos — endpoints de la E1.

  POST   /documents/upload   → US1 + US2 + US3
  GET    /documents/{id}     → US4
  GET    /documents/         → US5
"""
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, UploadFile, File, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.models.document import get_db
from app.services.document_service import upload_document, get_document, list_documents

router = APIRouter(prefix="/documents", tags=["documents"])


# ── Schemas de respuesta (Pydantic) ──────────────────────────────────────────

class DocumentResponse(BaseModel):
    id: str
    original_filename: str
    file_format: str
    file_size_bytes: int
    uploaded_at: datetime

    model_config = {"from_attributes": True}


class UploadResponse(BaseModel):
    message: str
    document: DocumentResponse


class DocumentListResponse(BaseModel):
    total: int
    documents: list[DocumentResponse]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/upload", response_model=UploadResponse, status_code=201)
async def upload_document_endpoint(
    file: UploadFile = File(..., description="Archivo PDF, DOCX o TXT"),
    db: Session = Depends(get_db),
):
    """
    **US1 + US2 + US3** — Sube un documento a la Knowledge Base.

    - Acepta: PDF, DOCX, TXT
    - Valida el formato antes de guardar
    - Retorna el ID único asignado al documento
    """
    doc = await upload_document(file, db)
    return UploadResponse(
        message="Documento subido exitosamente.",
        document=DocumentResponse.model_validate(doc),
    )


@router.get("/{doc_id}", response_model=DocumentResponse)
def get_document_endpoint(
    doc_id: str,
    db: Session = Depends(get_db),
):
    """
    **US4** — Recupera los metadatos de un documento por su ID.
    """
    doc = get_document(doc_id, db)
    return DocumentResponse.model_validate(doc)


@router.get("/", response_model=DocumentListResponse)
def list_documents_endpoint(
    skip: int = Query(default=0, ge=0, description="Registros a omitir (paginación)"),
    limit: int = Query(default=100, ge=1, le=500, description="Máximo de registros a retornar"),
    db: Session = Depends(get_db),
):
    """
    **US5** — Lista todos los documentos registrados en la Knowledge Base.

    Útil para el dashboard y el agente de recuperación (E2).
    """
    docs = list_documents(db, skip=skip, limit=limit)
    return DocumentListResponse(
        total=len(docs),
        documents=[DocumentResponse.model_validate(d) for d in docs],
    )
