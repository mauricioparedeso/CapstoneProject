"""
Router de documentos — endpoints de la E1.

  POST   /documents/upload   → US1 + US2 + US3
  GET    /documents/{id}     → US4
  GET    /documents/         → US5
"""
import time
from langchain_community.document_loaders import Docx2txtLoader, UnstructuredPowerPointLoader
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, UploadFile, File, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.models.document import get_db
from app.services.document_service import upload_document, get_document, list_documents

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.Chroma_Imp import vector_store
import shutil, os

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
    await file.seek(0)

    temp_path = f"temp_{doc.original_filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    indexing_msg = "No indexado (Formato no soportado para búsqueda)"

    try:
        if "pdf" in doc.file_format.lower():
            loader = PyPDFLoader(temp_path)
            pages = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(pages)
            for chunk in chunks:
                chunk.metadata["doc_id"] = doc.id
                chunk.metadata["source"] = doc.original_filename
            t0 = time.time()
            vector_store.add_documents(chunks)
            elapsed = round(time.time() - t0, 2)
            indexing_msg = f"Indexado exitosamente en {len(chunks)} fragmentos ({elapsed}s)"
            print(f"[indexing] PDF '{doc.original_filename}': {elapsed}s para {len(chunks)} chunks")

        elif "docx" in doc.file_format.lower():
            loader = Docx2txtLoader(temp_path)
            pages = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(pages)
            for chunk in chunks:
                chunk.metadata["doc_id"] = doc.id
                chunk.metadata["source"] = doc.original_filename
            t0 = time.time()
            vector_store.add_documents(chunks)
            elapsed = round(time.time() - t0, 2)
            indexing_msg = f"Indexado exitosamente en {len(chunks)} fragmentos ({elapsed}s)"
            print(f"[indexing] DOCX '{doc.original_filename}': {elapsed}s para {len(chunks)} chunks")

        elif "txt" in doc.file_format.lower():
            loader = TextLoader(temp_path)
            pages = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(pages)
            for chunk in chunks:
                chunk.metadata["doc_id"] = doc.id
                chunk.metadata["source"] = doc.original_filename
            t0 = time.time()
            vector_store.add_documents(chunks)
            elapsed = round(time.time() - t0, 2)
            indexing_msg = f"Indexado exitosamente en {len(chunks)} fragmentos ({elapsed}s)"
            print(f"[indexing] TXT '{doc.original_filename}': {elapsed}s para {len(chunks)} chunks")

        elif "pptx" in doc.file_format.lower():
            loader = UnstructuredPowerPointLoader(temp_path)
            pages = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(pages)
            for chunk in chunks:
                chunk.metadata["doc_id"] = doc.id
                chunk.metadata["source"] = doc.original_filename
            t0 = time.time()
            vector_store.add_documents(chunks)
            elapsed = round(time.time() - t0, 2)
            indexing_msg = f"Indexado exitosamente en {len(chunks)} fragmentos ({elapsed}s)"
            print(f"[indexing] PPTX '{doc.original_filename}': {elapsed}s para {len(chunks)} chunks")

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