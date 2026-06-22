FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data ./data
COPY src ./src
COPY evaluation ./evaluation

EXPOSE 8000 8501

CMD ["uvicorn", "src.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

