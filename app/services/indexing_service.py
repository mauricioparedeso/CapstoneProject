"""
Servicio de indexación — puente E1→E2.

Responsabilidad:
  - Leer el archivo físico guardado por E1
  - Dividirlo en chunks
  - Indexarlo en ChromaDB para que E3 pueda consultarlo

Se llama desde upload_document() en document_service.py
después de guardar el archivo y el registro en BD.
"""
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

from app.Chroma_Imp import vector_store

SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)


def _load_document(storage_path: str, file_format: str, original_filename: str):
    """
    Carga el contenido del archivo según su formato.
    Retorna una lista de Documents de LangChain.
    """
    path = str(storage_path)

    if file_format == "pdf":
        loader = PyPDFLoader(path)
    elif file_format == "docx":
        loader = Docx2txtLoader(path)
    elif file_format == "txt":
        loader = TextLoader(path, encoding="utf-8")
    else:
        raise ValueError(f"Formato no soportado para indexación: {file_format}")

    docs = loader.load()

    for doc in docs:
        doc.metadata["source"] = original_filename
        doc.metadata["doc_id"] = Path(storage_path).stem

    return docs


def index_document(storage_path: str, file_format: str, original_filename: str) -> int:
    """
    Indexa un documento en ChromaDB.

    Args:
        storage_path:      Ruta física del archivo en app/storage/
        file_format:       Formato del archivo (pdf, docx, txt)
        original_filename: Nombre original que subió el instructor

    Returns:
        Número de chunks indexados.
    """
    docs = _load_document(storage_path, file_format, original_filename)
    chunks = SPLITTER.split_documents(docs)

    if not chunks:
        raise ValueError(f"El archivo '{original_filename}' no tiene contenido indexable.")

    vector_store.add_documents(chunks)
    print(f"[indexing] '{original_filename}' indexado en {len(chunks)} chunks.")

    return len(chunks)