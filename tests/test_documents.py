"""
Tests de E1 — Document Ingestion Pipeline.

Cubre el DoD:
  ✅ Upload válido (PDF, DOCX, TXT)
  ✅ Formato inválido retorna error claro
  ✅ Persistencia: el documento queda guardado tras el upload
  ✅ Listing: se pueden listar documentos vía API

Correr con:
    pytest tests/test_documents.py -v
"""
import io
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.models.document import Base, get_db

# ── Base de datos en memoria para tests ──────────────────────────────────────

TEST_DATABASE_URL = "sqlite:///:memory:"
test_engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def override_get_db():
    Base.metadata.create_all(bind=test_engine)
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_file(content: bytes, filename: str, content_type: str):
    """Crea un objeto de archivo simulado para el test client."""
    return ("file", (filename, io.BytesIO(content), content_type))


# ── Tests: upload válido ──────────────────────────────────────────────────────

def test_upload_pdf_success():
    """US1 — Subir un PDF debe retornar 201 con ID del documento."""
    response = client.post(
        "/documents/upload",
        files=[make_file(b"%PDF-1.4 fake content", "syllabus.pdf", "application/pdf")],
    )
    assert response.status_code == 201
    data = response.json()
    assert data["message"] == "Documento subido exitosamente."
    assert "id" in data["document"]
    assert data["document"]["file_format"] == "pdf"
    assert data["document"]["original_filename"] == "syllabus.pdf"


def test_upload_docx_success():
    """US1 — Subir un DOCX debe retornar 201."""
    response = client.post(
        "/documents/upload",
        files=[make_file(
            b"PK fake docx content",
            "guide.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )],
    )
    assert response.status_code == 201
    assert response.json()["document"]["file_format"] == "docx"


def test_upload_txt_success():
    """US1 — Subir un TXT debe retornar 201."""
    response = client.post(
        "/documents/upload",
        files=[make_file(b"Plain text content", "notes.txt", "text/plain")],
    )
    assert response.status_code == 201
    assert response.json()["document"]["file_format"] == "txt"


# ── Tests: formato inválido ───────────────────────────────────────────────────

def test_upload_invalid_format_returns_400():
    """US2 — Un .exe debe retornar 400 con mensaje claro."""
    response = client.post(
        "/documents/upload",
        files=[make_file(b"binary content", "malware.exe", "application/octet-stream")],
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "exe" in detail
    assert "válidos" in detail.lower() or "soportado" in detail.lower()


def test_upload_invalid_format_png_returns_400():
    """US2 — Un .png debe retornar 400."""
    response = client.post(
        "/documents/upload",
        files=[make_file(b"\x89PNG fake", "image.png", "image/png")],
    )
    assert response.status_code == 400


# ── Tests: persistencia ───────────────────────────────────────────────────────

def test_upload_document_is_persisted():
    """US3 + US4 — Tras subir, el documento debe ser recuperable por ID."""
    # Subir
    upload_resp = client.post(
        "/documents/upload",
        files=[make_file(b"%PDF-1.4 content", "course.pdf", "application/pdf")],
    )
    assert upload_resp.status_code == 201
    doc_id = upload_resp.json()["document"]["id"]

    # Recuperar
    get_resp = client.get(f"/documents/{doc_id}")
    assert get_resp.status_code == 200
    doc = get_resp.json()
    assert doc["id"] == doc_id
    assert doc["original_filename"] == "course.pdf"


def test_get_nonexistent_document_returns_404():
    """US4 — Buscar un ID que no existe debe retornar 404."""
    response = client.get("/documents/id-que-no-existe")
    assert response.status_code == 404


# ── Tests: listing ────────────────────────────────────────────────────────────

def test_list_documents_returns_all():
    """US5 — El endpoint de listado debe incluir los documentos subidos."""
    # Subir dos documentos
    client.post(
        "/documents/upload",
        files=[make_file(b"content a", "a.txt", "text/plain")],
    )
    client.post(
        "/documents/upload",
        files=[make_file(b"content b", "b.txt", "text/plain")],
    )

    list_resp = client.get("/documents/")
    assert list_resp.status_code == 200
    data = list_resp.json()
    assert "total" in data
    assert "documents" in data
    assert data["total"] >= 2
