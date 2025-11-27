FROM python:3.12

WORKDIR /opt/src

ENV PYTHONPATH=/opt/src

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libssl-dev \
    libffi-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY ml-serve/requirements.txt requirements.txt

RUN  \
     pip install --upgrade pip setuptools wheel \
     && pip install --no-cache-dir -r requirements.txt 

RUN pip install aiohttp

COPY ml-serve/ .

EXPOSE 81

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "81"]