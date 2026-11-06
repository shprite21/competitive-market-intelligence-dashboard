FROM python:3.12-slim

WORKDIR /app

# Hosted containers fail closed until APP_AUTH_PASSWORD is configured.
ENV APP_AUTH_REQUIRED=true

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
