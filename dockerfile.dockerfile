FROM python:3.10-slim

# Create a non-root user to run the application
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Set proper permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Make port 7860 available to the world outside this container
EXPOSE 7860

# Define environment variable
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.enableCORS=false"]