# Use the official Python 3.9 image
FROM python:3.10.12

# Set the working directory to /code
WORKDIR /code

# Copy the current directory contents into the container at /code
COPY ./requirements.txt /code/requirements.txt

# Install ffmpeg and requirements.txt 
# Switch back to root to install packages
USER root
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user
# Switch to the "user" user
USER user
# Set home to the user's home directory
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user . $HOME/app


CMD ["uvicorn", "main:api", "--host", "0.0.0.0", "--port", "7860"]