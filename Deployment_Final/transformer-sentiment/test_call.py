import requests

# URL where your Flask API is running
url = "http://127.0.0.1:5000/predict"  # or http://192.168.2.159:5000 if testing from another device

# Input payload
data = {
    "text": "The movie was fantastic and emotionally powerful."
}

# Send POST request
response = requests.post(url, json=data)

# Show result
print("Status Code:", response.status_code)
print("Prediction Result:", response.json())
