# Use a specific working Python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create working directory
WORKDIR /app2

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy all code to container
COPY . .

# Expose port and run app
CMD ["gunicorn", "app2:app", "--bind", "0.0.0.0:8000"]
