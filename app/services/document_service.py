"""
Servicio de documentos — lógica de negocio para E1.

Cubre:
  - US1: Guardar archivo en disco y registrar en BD
  - US2: Validar formato del archivo
  - US3: Persistir registro y metadatos
  - US4: Recuperar un registro por ID
  - US5: Listar todos los registros
  + Puente E1→E2: indexar en ChromaDB tras el upload
"""
import uuid
from pathlib import Path

from fastapi import UploadFile, HTTPException
from sqlalchemy.orm import Session

from app.models.document import Document

# Formatos permitidos (US2)
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt", "pptx"}
ALLOWED_CONTENT_TYPES = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "text/plain": "txt",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
}

# Directorio de almacenamiento — separado de chroma_data
STORAGE_DIR = Path("app/storage")
STORAGE_DIR.mkdir(parents=True, exist_ok=True)


# ── US2: Validación ──────────────────────────────────────────────────────────

def _get_extension(filename: str) -> str:
    """Extrae la extensión del nombre de archivo en minúsculas."""
    return Path(filename).suffix.lstrip(".").lower()


def validate_file(file: UploadFile) -> str:
    """
    Valida formato del archivo.
    Retorna la extensión si es válida.
    Lanza HTTPException 400 si no lo es.
    """
    ext = _get_extension(file.filename or "")
    content_type = file.content_type or ""

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Formato '{ext}' no soportado. "
                f"Formatos válidos: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
            ),
        )

    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Content-Type '{content_type}' no reconocido para archivos de Knowledge Base."
            ),
        )

    return ext


# ── US1 + US3 + Puente E1→E2 ─────────────────────────────────────────────────

async def upload_document(file: UploadFile, db: Session) -> Document:
    """
    Lee el archivo, lo guarda en disco, crea el registro en BD
    e indexa el contenido en ChromaDB para que E3 pueda consultarlo.

    Returns:
        Document: el registro recién creado con su ID único.
    """
    # US2 — Validar antes de leer
    ext = validate_file(file)

    # Leer contenido
    content = await file.read()
    file_size = len(content)

    # Generar nombre único para el archivo en disco
    doc_id = str(uuid.uuid4())
    stored_filename = f"{doc_id}.{ext}"
    storage_path = STORAGE_DIR / stored_filename

    # US1 — Guardar archivo físico en app/storage/
    storage_path.write_bytes(content)

    # US3 — Crear registro en BD
    doc = Document(
        id=doc_id,
        filename=stored_filename,
        original_filename=file.filename or stored_filename,
        file_format=ext,
        file_size_bytes=file_size,
        storage_path=str(storage_path),
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    # Puente E1→E2 — Indexar en ChromaDB
    # Se hace después del commit para no perder el registro si la indexación falla
    try:
        from app.services.indexing_service import index_document
        index_document(
            storage_path=str(storage_path),
            file_format=ext,
            original_filename=doc.original_filename,
        )
    except Exception as e:
        # Si falla la indexación, el archivo igual queda guardado en E1
        print(f"[indexing] Advertencia: no se pudo indexar '{doc.original_filename}': {e}")

    return doc


# ── US4: Recuperar un documento ──────────────────────────────────────────────

def get_document(doc_id: str, db: Session) -> Document:
    """
    Busca un documento por su ID único.
    Lanza 404 si no existe.
    """
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail=f"Documento '{doc_id}' no encontrado.")
    return doc


# ── US5: Listar documentos ───────────────────────────────────────────────────

def list_documents(db: Session, skip: int = 0, limit: int = 100) -> list[Document]:
    """
    Retorna todos los documentos registrados, paginados.
    Ordenados por fecha de subida descendente.
    """
    return (
        db.query(Document)
        .order_by(Document.uploaded_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )