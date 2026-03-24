import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "age": 52,
    "sex": 1,
    "cp": 0,
    "trestbps": 125,
    "chol": 212,
    "fbs": 0,
    "restecg": 1,
    "thalach": 168,
    "exang": 0,
    "oldpeak": 1.0
}

response = requests.post(url, json=data)

print(response.json())