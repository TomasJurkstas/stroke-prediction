import random
from locust import HttpUser, task, between

test_applications = [{"age": 74, "avg_glucose_level": 120},
{"age": 5, "avg_glucose_level": 50},
{"age": 26, "avg_glucose_level": 70},
{"age": 58, "avg_glucose_level": 240},
{"age": 74, "avg_glucose_level": 42},
{"age": 80, "avg_glucose_level": 140},
{"age": 35, "avg_glucose_level": 82},
{"age": 43, "avg_glucose_level": 65},
{"age": 12, "avg_glucose_level": 80},]


class MyUser(HttpUser):
    wait_time = between(1, 5)
    host = 'https://default-service-l3vhobwizq-nw.a.run.app/'

    @task
    def make_prediction(self):
        headers = {"Content-Type": "application/json"}
        data = random.choice(test_applications)
        response = self.client.post("/predict", json=data, headers=headers)

        if response.status_code == 200:
            print("Prediction successful")
        else:
            print(f"Failed with status code: {response.status_code}")