# Use the official Python image as the base image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file to the working directory
COPY requirements.txt .

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential cmake libopenblas-dev liblapack-dev libx11-dev libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*
    
# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the working directory
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Set the environment variable for Flask
ENV FLASK_APP=app.py

# Set the environment variable for Flask's run mode
ENV FLASK_RUN_HOST=0.0.0.0

# Define the command to run the application using Flask's built-in server
CMD ["flask", "run"]
