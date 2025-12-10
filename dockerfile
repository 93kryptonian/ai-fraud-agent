# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=10000
EXPOSE 10000

CMD ["streamlit", "run", "ui/app.py", "--server.port=10000", "--server.address=0.0.0.0"]
