# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# This command runs in a separate layer and will be cached
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the function is listening on
EXPOSE 8080 

# Define the command to run the application
CMD ["functions-framework", "--target=consolidator", "--source=main.py"]
