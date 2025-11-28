FROM python:3.11-slim

# Workdir di container
WORKDIR /code

# Install dependency
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua file (api.py, model, dll)
COPY . .

# Hugging Face expect service di port 7860
ENV PORT=7860

# Jalankan Flask via gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:7860", "api:app"]
