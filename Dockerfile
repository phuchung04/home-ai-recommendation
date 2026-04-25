FROM python:3.11-slim

WORKDIR /app

# Cài đặt dependencies hệ thống
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements trước để tận dụng Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ ứng dụng
COPY . .

# Expose port 8000
EXPOSE 8000

# Command chạy ứng dụng - ✅ Thay đổi ở đây
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]