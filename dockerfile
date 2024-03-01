# Use the official Python image as the base image
FROM python:3.11

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install dependencies from the requirements file
RUN pip install --no-cache-dir -r requirements.txt

# Copy the contents of the local /api directory to the /app directory in the container
COPY /api /app

# Specify the command to run on container start
CMD ["python", "main.py"]
