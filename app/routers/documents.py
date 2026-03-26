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

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.Chroma_Imp import vector_store # Importamos el store que configuraste antes
import shutil, os

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
    indexing_status: str # Nuevo campo para confirmar ChromaDB


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
    # 1. Guardar metadatos en SQL
    doc = await upload_document(file, db)
    
    await file.seek(0) 
    # 2. PROCESAMIENTO PARA CHROMADB (E2/E3)
    # Guardamos temporalmente para que los Loaders de LangChain puedan leerlo
    temp_path = f"temp_{doc.original_filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    indexing_msg = "No indexado (Formato no soportado para búsqueda)"
    
    try:
        # Elegir el cargador según el formato
        if "pdf" in doc.file_format.lower():
            loader = PyPDFLoader(temp_path)
            pages = loader.load()
            # Fragmentar el texto (Chunking)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(pages)
            
            # Inyectar metadatos adicionales para el agente
            for chunk in chunks:
                chunk.metadata["doc_id"] = doc.id
                chunk.metadata["source"] = doc.original_filename
            
            # GUARDAR EN CHROMA (Esto llena el sqlite3 y genera vectores)
            vector_store.add_documents(chunks)
            indexing_msg = f"Indexado exitosamente en {len(chunks)} fragmentos"
            
    except Exception as e:
        indexing_msg = f"Error en indexación: {str(e)}"
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

        
    return UploadResponse(
        message="Documento subido exitosamente.",
        document=DocumentResponse.model_validate(doc),
        indexing_status=indexing_msg
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
