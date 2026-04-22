# Imagen base
FROM python:3.11-slim

# Evita problemas de buffer
ENV PYTHONUNBUFFERED=1

# Directorio de trabajo
WORKDIR /app

# Copiar dependencias primero (mejor cache)
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt
RUN sudo apt-get install -y libenchant-2-2
RUN sudo apt-get install -y aspell-es

# Copiar el resto del código
COPY . .

# Exponer puertos (informativo)
EXPOSE 8000
EXPOSE 8501

# Comando por defecto (se sobreescribe en docker-compose)
CMD ["python"]