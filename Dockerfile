FROM python:3.11-slim

WORKDIR /app

# Copy the application code first
COPY . .

# Copy the pre-installed dependencies from the build workspace into the correct location
# The trailing slash on packages/ is important to copy the contents.
COPY packages/ /usr/local/lib/python3.11/site-packages/

EXPOSE 8080

CMD ["functions-framework", "--target=consolidator", "--source=main.py"]