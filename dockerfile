#wsl install Ubuntu -> Config Docker
#abrir Ubuntu -> Docker run hello-world

#instalar wsl en vscode

#Montar en documents el git pull
#cd /mnt/c/Users/olakh/Documents/CapstoneProject

#Docker compose up --build

#Si ya está construído
#Docker compose up -d

#docker compose logs -f

#Docker compose stop

#docker exec fastapi_app pip install langfuse

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