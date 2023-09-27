# Start from the pytorch image
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
# FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Install system tools, build tools, and git
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl

# Copy requirements.txt and install the dependencies
COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install rerun
RUN pip3 install git+https://github.com/rerun-io/rerun.git@main#subdirectory=rerun_py

# Copy main.py into the Docker image
COPY huggingface_pipeline.py .
COPY main.py .

# CMD ["python3", "main.py"]
