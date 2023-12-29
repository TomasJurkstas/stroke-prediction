# Start from a base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements_docker.txt requirements_docker.txt
COPY static static

# Install the required packages
RUN pip install --upgrade pip
RUN pip install -r requirements_docker.txt

# Copy the application code into the container
COPY ["logisticregression.pkl", "app.py", "./"]

# Expose the app port
EXPOSE 80

# Run command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]