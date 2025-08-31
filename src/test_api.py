import requests

# URL of your running FastAPI server
API_URL = "http://127.0.0.1:5000/predict"

# Example passenger data
sample_input = {
    "Pclass": 3,
    "Sex": 1,       # 0 = female, 1 = male
    "Age": 29.0,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": 0   # 0 = C, 1 = Q, 2 = S (adjust as per your encoding)
}

response = requests.post(API_URL, json=sample_input)

if response.status_code == 200:
    print("✅ Prediction received:", response.json())
else:
    print("❌ Request failed:", response.status_code, response.text)
