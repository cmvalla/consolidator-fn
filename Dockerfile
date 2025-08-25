FROM python:3.11-slim

WORKDIR /app

# This build argument will be the name of the cache file from the GCS bucket.
# It defaults to a non-existent file to handle the first run case.
ARG CACHE_FILE=pip-cache.tar.gz

# Copy the cache file and extract it. The shell will handle the file-not-found case.
COPY ${CACHE_FILE} /tmp/cache.tar.gz
RUN tar -xzf /tmp/cache.tar.gz -C /root/.cache || echo "No cache file found."

# Copy only the requirements file to leverage Docker layer caching.
COPY requirements.txt .

# Install dependencies. By NOT using --no-cache-dir, pip will create and use /root/.cache/pip.
RUN pip install -r requirements.txt

# Copy the rest of the application code.
COPY . .

# Expose the port the function is listening on.
EXPOSE 8080

# Set the entrypoint for the function.
CMD ["functions-framework", "--target=consolidator", "--source=main.py"]
