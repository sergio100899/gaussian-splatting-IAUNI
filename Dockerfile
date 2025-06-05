# FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# Establecer el directorio de trabajo
WORKDIR /app

# Instala herramientas necesarias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libaio1 \
    libglib2.0-0 \
    libgtk-3-0 \
    libnotify-dev \
    libnss3 \
    libxss1 \
    libasound2 \
    wget \
    unzip \
    git \
    curl \
    make \
    python3 \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Instalar dependencias de Python
RUN pip install --upgrade pip
RUN pip install uv

# Copiar el archivo requirements.txt al directorio de trabajo
COPY pyproject.toml .
COPY uv.lock .

# Instalar las dependencias de Python
# RUN uv sync

# Copiar el resto del código al directorio de trabajo
COPY . .

RUN uv venv

RUN uv pip install ".[torch]"
RUN uv pip install ".[base]" 
# RUN uv pip install --no-build-isolation ".[submodules]"

#Descargar dataset
RUN wget https://huggingface.co/camenduru/gaussian-splatting/resolve/main/tandt_db.zip && \
    unzip tandt_db.zip

# RUN source .venv/bin/activate

# Exponer el puerto 8000
# EXPOSE 8000

ENV PYTHONUNBUFFERED=1

# Ejecutar la aplicación
CMD ["tail", "-f", "/dev/null"]