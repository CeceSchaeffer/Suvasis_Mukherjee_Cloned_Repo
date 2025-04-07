FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY data/ ./data/

RUN mkdir -p output

EXPOSE 8505

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8505", "--server.address=0.0.0.0"]
