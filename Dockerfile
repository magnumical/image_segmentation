# Use the official Python 3.12 image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Update the package lists
RUN apt-get update

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Command to run the Python script with arguments passed during container runtime
ENTRYPOINT ["python", "inference.py"]
