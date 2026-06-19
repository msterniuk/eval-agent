from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

class ModelParams(BaseModel):
    model_id: str

class PromptRequest(BaseModel):
    prompt: str
    model_params: ModelParams

class GeminiRequestHandler:
    def __init__(self, model):
        self.model = model

    def generate_response(self, prompt: str, model_params: dict):
        try:
            response = self.model.generate_content(prompt, labels={"model_id": model_params.get("model_id")})
            return response
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

# Mock model class for demonstration purposes
class MockModel:
    def generate_content(self, prompt, labels):
        return f"Generated content for prompt: {prompt} with model_id: {labels['model_id']}"

# Initialize the handler with a mock model
handler = GeminiRequestHandler(MockModel())

@app.post("/generate")
def generate_content(request: PromptRequest):
    response = handler.generate_response(request.prompt, request.model_params.dict())
    return {"response": response}

# To run the application, use the following command:
# uvicorn your_script_name:app --reload

#http POST http://127.0.0.1:8000/generate prompt="Hello" model_params:='{"model_id": "test"}'
#curl -X POST "http://127.0.0.1:8000/generate" -H "Content-Type: application/json" -d '{"prompt": "Hello", "model_params": {"model_id": "test"}}'