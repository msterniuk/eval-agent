from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import vertexai
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
            # response_schema=response_schema
        )
        self.model = GenerativeModel(
            model_name="gemini-1.5-flash-001",
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
            # Assuming generate_content is a method of the model
            response = self.generate_content(prompt, labels=labels)
            return response
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def generate_content(self, prompt, labels):
        # Mock response for demonstration purposes
        return f"Generated content for prompt: {prompt} with model_id: {labels['model_id']}"

# Initialize the handler with the specific model id
handler = GeminiRequestHandler("gemini-1.5-flash-001")

@app.post("/generate")
def generate_content(request: PromptRequest):
    prompt = '[{"sample_id": "g-ChdDSUhNMG9nS0VJQ0FnTUNBcy1XOXZ3RRAB", "review_text": "overall score 5 fantastic stay at park plaza chennai omr! the location was perfect for exploring the city just short walk to major attractions and vibrant nightlife. the rooms were modern spotlessly clean and the beds were incredibly comfortable and like overall service of the staff so can one can visit this fantastic hotel."}, {"sample_id": "9eb2a4b01e10b1cfb89cd77cd4df9b1ce974cb2e", "review_text": "overall score 5 nuestra estancia en el iberostar bella vista fue simplemente maravillosa. desde el momento en que llegamos nos sentimos bienvenidos. indira la animadora hizo que cada día fuera especial con su energía contagiosa su amabilidad. volveremos pronto! hotel highlights great view"}]Follow JSON schema.<JSONSchema>"{\"description\":\"classifying hotel customer review focusing on hotel topics.\",\"items\":{\"properties\":{\"Nested\":{\"description\":\"Extract classification mentioned in the customer review\",\"items\":{\"additionalProperties\":false,\"properties\":{\"Aspect\":{\"description\":\"aspect related to the subtopic (e.g. quality, quantity)\",\"type\":\"string\"},\"Emotion\":{\"description\":\"Emotion related to the subtopic\",\"type\":\"string\"},\"Sentiment\":{\"description\":\"Sentiment related to the subtopic (e.g. POSITIVE, NEUTRAL, NEGATIVE)\",\"type\":\"string\"},\"Subtopic\":{\"description\":\"subtopic related to the topic\",\"type\":\"string\"},\"Topic\":{\"description\":\"topic mentioned in the review\",\"type\":\"string\"}},\"required\":[\"sample_id\",\"Topic\",\"Subtopic\",\"Aspect\",\"Sentiment\",\"Emotion\"],\"type\":\"object\"},\"type\":\"array\"},\"sample_id\":{\"description\":\"sample_id\",\"type\":\"string\"}},\"type\":\"array\"},\"title\":\"Classify topic review\",\"type\":\"object\"}"</JSONSchema>'
    response = handler.generate_response(prompt, labels={"model_id": request.model_params.model_id})
    
    return {"response": response}

# To run the application, use the following command:
# uvicorn your_script_name:app --reload