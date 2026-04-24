FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instalar dependencias del sistema (BIEN HECHO)
RUN apt-get update && apt-get install -y \
    libenchant-2-2 \
    aspell-es \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero (cache)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY . .

EXPOSE 8000
EXPOSE 8501

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--reload"]