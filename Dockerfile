FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the virtual environment, which contains all dependencies
COPY venv /app/venv

# Copy the application code
COPY . .

# Add the virtual environment's bin to the PATH
# This ensures that any shell running in the container can find the executables.
ENV PATH="/app/venv/bin:$PATH"

EXPOSE 8080

# Run the functions-framework from the virtual environment
CMD ["functions-framework", "--target=consolidator", "--source=main.py"]
