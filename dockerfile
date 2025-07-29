# Use a slim Python base image for a smaller footprint
FROM python:3.9-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy only the requirements file first to leverage Docker's build cache
# This means if only your code changes, but requirements.txt doesn't,
# Docker won't re-install all dependencies every time you build.
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir: Prevents pip from storing downloaded packages in a cache,
#                 reducing the image size.
# -r: Install from the requirements file.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python application code and the .env file into the container
# IMPORTANT: For production, you would typically use a secrets management
# system and NOT copy .env directly into the final image.
# For local development and quick testing, this is acceptable.
COPY app.py .
COPY .env .

# Expose the port Streamlit uses (default is 8501)
EXPOSE 8501

# Define the command to run the Streamlit application
# 0.0.0.0 allows access from outside the container (e.g., your host machine)
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]