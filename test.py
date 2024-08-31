import requests
import json

# Base URL of your API
BASE_URL = "https://rohit4242-transcripter--different-meta-llama-3-8b-instru-122315.modal.run"  # Update this with the actual URL of your API

def test_summarize():
    endpoint = f"{BASE_URL}/summarize"
    data = {
        "text": "Your sample text here. This can be a longer piece of content that you want to process."
    }
    response = requests.post(endpoint, json=data)
    print("Summarize Endpoint:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_important_questions():
    endpoint = f"{BASE_URL}/important-questions"
    data = {
        "text": "Your sample text here. This can be a longer piece of content that you want to process."
    }
    response = requests.post(endpoint, json=data)
    print("Important Questions Endpoint:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_generate_quiz():
    endpoint = f"{BASE_URL}/generate-quiz"
    data = {
        "text": "Your sample text here. This can be a longer piece of content that you want to process."
    }
    response = requests.post(endpoint, json=data)
    print("Generate Quiz Endpoint:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_generate_web():
    endpoint = f"{BASE_URL}/generate_web"
    data = {
        "prompts": ["Tell me a short story about a robot."],
        "settings": None
    }
    response = requests.post(endpoint, json=data)
    print("Generate Web Endpoint:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

if __name__ == "__main__":
    test_summarize()
    test_important_questions()
    test_generate_quiz()
    test_generate_web()