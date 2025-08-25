FROM python:3.11-slim

WORKDIR /app

# First, copy only the requirements file.
COPY requirements.txt .

# Install the dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Now, copy the rest of the application code.
COPY . .

# Expose the port the function is listening on.
EXPOSE 8080

# Set the entrypoint for the function.
CMD ["functions-framework", "--target=consolidator", "--source=main.py"]