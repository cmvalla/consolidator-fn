FROM python:3.11-slim

WORKDIR /app

# Add the vendored packages directory to Python's path
ENV PYTHONPATH /app/packages

# Copy the application code first
COPY . .

# Copy the pre-installed dependencies from the build workspace
COPY packages /app/packages

EXPOSE 8080

# The entrypoint will now use the Python interpreter with the correct path
CMD ["python3", "-m", "functions_framework", "--target=consolidator", "--source=main.py"]