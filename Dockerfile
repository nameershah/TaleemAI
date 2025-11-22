# Use a slim Python image as a base
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the dependencies file first to leverage Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the correct port for Streamlit
EXPOSE 8501

# FIX: Set environment variable to disable XSRF protection for file uploads 
# on Hugging Face Docker Spaces (prevents the 403 AxiosError)
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# Set the entrypoint to run the Streamlit app
# Note: --server.port 8501 and --server.address 0.0.0.0 are critical for HF Spaces
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]