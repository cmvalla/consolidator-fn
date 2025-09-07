# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the rest of the application code
COPY . .

# Expose the port the function is listening on
EXPOSE 8080 

# Define the command to run the application
ENTRYPOINT ["python", "-m", "functions_framework", "--target", "consolidator", "--port", "8080"]
