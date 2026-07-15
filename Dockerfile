FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app/reasoning_engine
COPY . .

RUN pip install --no-cache-dir -e .[dev]

CMD ["python"]
