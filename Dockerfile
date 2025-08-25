FROM python:3.11-slim

WORKDIR /app

# This build argument will be the name of the cache file from the GCS bucket
ARG CACHE_FILE

# Copy the cache file if it exists and extract it. Fail gracefully if it does not.
# This populates /root/.cache/pip
COPY . /app
RUN if [ -f "/app/${CACHE_FILE}" ]; then \
    echo "Cache file found, extracting..."; \
    tar -xzf "/app/${CACHE_FILE}" -C /root/.cache; \
    else \
    echo "No cache file found, proceeding without cache."; \
    fi

# Now, install dependencies. pip will use the restored cache if it was extracted.
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the function is listening on.
EXPOSE 8080

# Set the entrypoint for the function.
CMD ["functions-framework", "--target=consolidator", "--source=main.py"]