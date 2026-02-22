# Use a lightweight, standard Python environment
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# IMPORTANT: LightGBM requires the libgomp1 system library to run on Linux servers!
# If this was missing on the platform, it might be why your submission failed.
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# 1. Copy and install requirements
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# 2. Copy your actual submission files
COPY solution.py .
COPY model.joblib .

# 3. Copy the testing script and test data
COPY emulate_judge.py .
COPY local_val.csv . 

# Run the judge script when the container starts
CMD ["python", "emulate_judge.py"]
