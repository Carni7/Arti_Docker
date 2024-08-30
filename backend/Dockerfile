FROM python:3.11.5-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy the Flask app
COPY . .

EXPOSE 5000

# Run the app
CMD ["python", "Endpoints.py"]
