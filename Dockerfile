FROM tensorflow/tensorflow:latest

# Set the working directory inside the container
WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip3 install --ignore-installed --no-cache-dir -r requirements.txt

# Copy the Flask application files into the container
COPY server.py .
COPY model.keras .

# Expose the port the Flask app runs on
EXPOSE 5000

# Define the command to run the Flask app when the container starts
CMD ["python3", "server.py"]