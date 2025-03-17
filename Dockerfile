# First stage: Build dependencies in a lightweight image
FROM python:3.10-slim AS builder

WORKDIR /app

# Copy only requirements to leverage Docker caching
COPY requirements.txt .

# Install dependencies in a virtual environment
RUN python -m venv /venv && \
    /venv/bin/pip install --no-cache-dir -r requirements.txt

# Second stage: Create the final lightweight image
FROM python:3.10-slim

WORKDIR /app

# Copy the pre-installed dependencies from builder stage
COPY --from=builder /venv /venv

# Copy application files
COPY . .

# Use the pre-installed environment
ENV PATH="/venv/bin:$PATH"

# Expose port for Render
EXPOSE 5000

# Set default command to run with Gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]