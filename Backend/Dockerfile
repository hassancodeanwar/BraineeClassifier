# Use Python 3.10.12 as the base image
FROM python:3.10.12-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create uploads directory
RUN mkdir -p uploads

# Download the model file (you'll need to provide the model file or download link)
# Uncomment and modify the following line if you need to download the model
# RUN wget -O model/Brain_Tumor_Classifier_Enhanced.h5 [MODEL_DOWNLOAD_URL]

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable to run Flask in production
ENV FLASK_ENV=production

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "main:app"]
