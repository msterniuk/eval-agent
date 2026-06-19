import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from google.cloud import aiplatform
from vertexai import generative_models
from vertexai.generative_models import GenerativeModel, GenerationConfig, SafetySetting

app = FastAPI()

class ModelParams(BaseModel):
    model_id: str
    output_schema: Optional[Dict[str, Any]] = None

class PromptRequest(BaseModel):
    prompt: str
    model_params: ModelParams

class GeminiRequestHandler:
    def __init__(self, model_id):
        self.model_id = model_id
        generation_config = GenerationConfig(
            temperature=0.5,
            top_p=0.9,
            top_k=40,
            presence_penalty=0.5,
            frequency_penalty=0.5,
            response_mime_type="application/json",
        )
        self.model = GenerativeModel(
            model_name=model_id,
            system_instruction="Generate a response based on the provided prompt.",
            generation_config=generation_config,
            safety_settings=self._get_safety_settings()
        )

    def _get_safety_settings(self):
        """Returns safety settings for generative models to block unsafe content."""
        categories = [
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT,
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT
        ]
        return [SafetySetting(category=cat, threshold=generative_models.HarmBlockThreshold.BLOCK_NONE) for cat in categories]

    def generate_response(self, prompt: str, labels: dict):
        try:
            response = self.model.generate_content(prompt, labels=labels)
            return response.to_dict()  # Convert the response to a dictionary
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

# Initialize the handler with the specific model id
handler = GeminiRequestHandler("gemini-1.5-flash-001")

@app.post("/generate")
def generate_content(request: PromptRequest):
    response = handler.generate_response(request.prompt, labels={"model_id": request.model_params.model_id})
    return {"response": response}



if __name__ == "__main__":
    uvicorn.run("txt_analy_api_v5:app", host="127.0.0.1", port=5000, reload=True)

# To run the application, use the following command:
# uvicorn your_script_name:app --reload