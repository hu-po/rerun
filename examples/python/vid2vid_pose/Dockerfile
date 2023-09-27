# Start from the pytorch image
#FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
RUN pip install --upgrade pip
RUN python --version

# Copy requirements.txt and install the dependencies
COPY requirements_diff.txt .
RUN pip install -r requirements_diff.txt

# Copy main.py into the Docker image
COPY main.py .

CMD ["python3", "main_diff.py"]
