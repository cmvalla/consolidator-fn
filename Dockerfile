FROM python:3.11-slim

WORKDIR /app

# This build argument will be the name of the cache file from the GCS bucket
ARG CACHE_FILE

# Copy the cache file into the image and extract it to the location pip uses.
# This will populate /root/.cache/pip. Fail gracefully if the cache doesn't exist.
COPY ${CACHE_FILE} /tmp/cache.tar.gz
RUN tar -xzf /tmp/cache.tar.gz -C /root/.cache || echo "No cache found or cache is invalid."

# Now, copy the requirements file and install dependencies.
# pip will use the restored cache in /root/.cache/pip, speeding up the process.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the function is listening on.
EXPOSE 8080

# Set the entrypoint for the function.
CMD ["functions-framework", "--target=consolidator", "--source=main.py"]
