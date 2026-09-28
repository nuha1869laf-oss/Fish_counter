import os
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

load_dotenv()

api_key = os.getenv("ROBOFLOW_API_KEY")

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=api_key
)
image_path = "uploads/clownfish.webp"
model_id = "nuha-ymzrs/fish-detection-fuc8i-4-rfdetr-small-t1"
result = client.infer(
    image_path,
    model_id=model_id
)
print(result)