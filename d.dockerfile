# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.8-slim

WORKDIR /app

# Copy local code to the container image.
COPY . /app

# Install production dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Creating a directory for our application's logs
RUN mkdir /var/log/flask-server-llm-studio

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For more detaisl you can check Flask deployment options.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app