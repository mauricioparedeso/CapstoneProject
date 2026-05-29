"""
Router de documentos — endpoints de la E1.

  POST   /documents/upload   → US1 + US2 + US3
  GET    /documents/{id}     → US4
  GET    /documents/         → US5
"""
from datetime import datetime

from fastapi import APIRouter, Depends, UploadFile, File, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.models.document import get_db
from app.services.document_service import upload_document, get_document, list_documents
from app.Chroma_Imp import vector_store
import os

router = APIRouter(prefix="/documents", tags=["documents"])

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
    indexing_status: str

class DocumentListResponse(BaseModel):
    total: int
    documents: list[DocumentResponse]

@router.post("/upload", response_model=UploadResponse, status_code=201)
async def upload_document_endpoint(
    file: UploadFile = File(..., description="Archivo PDF, DOCX, TXT o PPTX"),
    db: Session = Depends(get_db),
):
    doc = await upload_document(file, db)
    indexing_msg = "Indexado exitosamente en ChromaDB"

    return UploadResponse(
        message="Documento subido exitosamente.",
        document=DocumentResponse.model_validate(doc),
        indexing_status=indexing_msg
    )

@router.get("/{doc_id}", response_model=DocumentResponse)
def get_document_endpoint(doc_id: str, db: Session = Depends(get_db)):
    doc = get_document(doc_id, db)
    return DocumentResponse.model_validate(doc)

@router.get("/", response_model=DocumentListResponse)
def list_documents_endpoint(
    skip: int = Query(default=0, ge=0, description="Registros a omitir (paginación)"),
    limit: int = Query(default=100, ge=1, le=500, description="Máximo de registros a retornar"),
    db: Session = Depends(get_db),
):
    docs = list_documents(db, skip=skip, limit=limit)
    return DocumentListResponse(
        total=len(docs),
        documents=[DocumentResponse.model_validate(d) for d in docs],
    )

@router.delete("/{doc_id}", status_code=200)
def delete_document_endpoint(doc_id: str, db: Session = Depends(get_db)):
    from app.services.document_service import get_document
    from app.models.document import Document

    doc = get_document(doc_id, db)

    try:
        results = vector_store.get(where={"doc_id": doc_id})
        if results and results.get("ids"):
            vector_store.delete(ids=results["ids"])
    except Exception as e:
        print(f"[WARNING] No se pudieron eliminar chunks de ChromaDB: {e}")

    storage_path = f"app/storage/{doc_id}.{doc.file_format}"
    if os.path.exists(storage_path):
        os.remove(storage_path)

    db.delete(doc)
    db.commit()

    return {"message": f"Documento {doc_id} eliminado correctamente."}