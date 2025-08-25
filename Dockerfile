FROM python:3.11-slim

WORKDIR /app

# Copy pre-installed dependencies from the Cloud Build workspace
# The trailing slash on `packages/` is important!
COPY packages/ /usr/local/lib/python3.11/site-packages/

# Copy the application code
COPY . .

# Expose the port the function is listening on
EXPOSE 8080

# Set the entrypoint for the function
CMD ["functions-framework", "--target=consolidator", "--source=main.py"]
