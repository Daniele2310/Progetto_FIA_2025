FROM python:3.13-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app
RUN mkdir -p /app/results
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "main.py"]
