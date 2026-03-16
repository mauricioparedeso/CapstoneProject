"""
Modelo de base de datos para documentos de la Knowledge Base.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, DateTime, Integer, create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# --- DB setup ---
DATABASE_URL = "sqlite:///./knowledge_base.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class Document(Base):
    """Registro de un documento subido a la Knowledge Base."""

    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    file_format = Column(String, nullable=False)   # pdf | docx | txt
    file_size_bytes = Column(Integer, nullable=False)
    storage_path = Column(String, nullable=False)
    uploaded_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


def create_tables():
    """Crea las tablas si no existen."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency de FastAPI para inyectar sesión de BD."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
