# Knowledge Base Curator - E1: Document Ingestion Pipeline

API REST para subir, validar y gestionar documentos de la Knowledge Base del sistema de análisis curricular con IA.

---

## Requisitos

- Python 3.10+
- pip

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/mauricioparedeso/CapstoneProject.git
cd CapstoneProject

# 2. Instalar dependencias
pip install -r requirements.txt
```

---

## Levantar la API

Desde la raíz del proyecto:

```bash
uvicorn app.main:app --reload
```

La API queda disponible en: `http://localhost:8000`

> El flag `--reload` hace que la API se reinicie automáticamente cada vez que guardas un cambio en el código. Útil en desarrollo.

---

## Documentación interactiva

Con la API corriendo, abre el browser en:

```
http://localhost:8000/docs
```

Desde ahí puedes probar todos los endpoints visualmente, sin necesidad de usar Postman ni escribir código.

---

## Endpoints disponibles

### `POST /documents/upload`
Sube un documento a la Knowledge Base.

**Formatos aceptados:** PDF, DOCX, TXT

Ejemplo de respuesta exitosa (201):
```json
{
  "message": "Documento subido exitosamente.",
  "document": {
    "id": "5ed5e5eb-a312-4b4b-a90d-70dac8a113aa",
    "original_filename": "syllabus.pdf",
    "file_format": "pdf",
    "file_size_bytes": 204800,
    "uploaded_at": "2026-03-15T14:30:00"
  }
}
```

Ejemplo de error por formato inválido (400):
```json
{
  "detail": "Formato 'exe' no soportado. Formatos válidos: docx, pdf, txt."
}
```

---

### `GET /documents/`
Lista todos los documentos registrados en la Knowledge Base.

Parámetros opcionales de paginación:
- `skip` - registros a omitir (default: 0)
- `limit` - máximo de registros a retornar (default: 100)

Ejemplo: `GET /documents/?skip=0&limit=10`

---

### `GET /documents/{id}`
Retorna los metadatos de un documento específico por su ID.

Ejemplo: `GET /documents/5ed5e5eb-a312-4b4b-a90d-70dac8a113aa`

---

### `GET /health`
Verifica que la API está corriendo.

```json
{ "status": "ok", "version": "0.1.0" }
```

---

## Estructura del proyecto

```
CapstoneProject/
├── app/
│   ├── main.py                    # FastAPI app, registro de rutas
│   ├── models/
│   │   └── document.py            # Modelo SQLAlchemy + configuración de BD
│   ├── routers/
│   │   └── documents.py           # Endpoints REST
│   ├── services/
│   │   └── document_service.py    # Lógica de negocio (upload, validación, queries)
│   └── storage/                   # Archivos físicos subidos (ignorado en git)
├── tests/
│   └── test_documents.py          # Tests automatizados de E1
├── Examples/                      # Scripts pedagógicos de LangGraph
├── requirements.txt
└── README.md
```

---

## Almacenamiento

- **Archivos físicos:** se guardan en `app/storage/` con un UUID como nombre (ej. `5ed5e5eb-a312-4b4b-a90d-70dac8a113aa.pdf`). El nombre original queda registrado en la base de datos.
- **Base de datos:** SQLite local (`knowledge_base.db` en la raíz). Guarda nombre original, formato, tamaño, ruta de almacenamiento y fecha de subida por cada documento.

> En producción, `app/storage/` se reemplazaría por un servicio de almacenamiento en la nube (S3, GCS, etc.) y SQLite por PostgreSQL. Solo cambia el `DATABASE_URL` en `app/models/document.py`.

---

## Correr los tests

```bash
pytest tests/test_documents.py -v
```

Los tests usan una base de datos en memoria y no requieren tener la API corriendo. Cubren:

- Upload válido (PDF, DOCX, TXT)
- Rechazo de formatos inválidos con mensaje claro
- Persistencia: el documento queda recuperable tras el upload
- Listado de documentos via API

---

## Dependencias principales

| Librería | Uso |
|---|---|
| `fastapi` | Framework web para la API REST |
| `uvicorn` | Servidor ASGI para correr FastAPI |
| `sqlalchemy` | ORM para la base de datos |
| `python-multipart` | Necesario para recibir archivos en FastAPI |
| `pypdf` | Lectura de PDFs (usado en E2) |
| `python-docx` | Lectura de DOCX (usado en E2) |
| `chromadb` | Base de datos vectorial (usado en E2) |
| `langgraph` | Orquestación del agente de IA (usado en E2+) |

---

## Epicas del proyecto

| Epica | Descripcion | Estado |
|---|---|---|
| E1 | Document Ingestion Pipeline | Sprint 1 - en progreso |
| E2 | Retrieval System | Backlog |
| E3 | AI Analysis | Backlog |
| E4 | Instructor Dashboard | Backlog |
| E5 | Observability y Feedback | Backlog |
