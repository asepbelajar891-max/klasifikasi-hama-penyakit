# Gunakan Python 3.10 sebagai base image
FROM python:3.10-slim-buster

# Set working directory di dalam container
WORKDIR /app

# Copy requirements.txt dan install dependensi
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh isi project ke dalam container
COPY . .

# Expose port yang digunakan Flask (default 5000)
EXPOSE 5000

# Perintah untuk menjalankan aplikasi menggunakan Gunicorn
# Sesuaikan jumlah worker dan thread sesuai kebutuhan server Anda
# Misalnya, untuk server dengan 2 CPU core, Anda bisa coba --workers 2 --threads 4
# Untuk GPU, pastikan driver dan CUDA terinstal di base image atau gunakan image TensorFlow yang sudah mendukung GPU.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]