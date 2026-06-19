from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import json

app = FastAPI()

class ModelParams(BaseModel):
    model_id: str
    output_schema: Optional[Dict[str, Any]] = None

class PromptRequest(BaseModel):
    prompt: str
    model_params: ModelParams

class GeminiRequestHandler:
    def __init__(self, model):
        self.model = model

    def generate_response(self, prompt: str, labels: dict):
        try:
            response = self.model.generate_content(prompt, labels=labels)
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
    json_data = [{"sample_id": "168c8f7c4c2a687f8f2c632ad10b83ab697be27f", "review_text": "overall score 1 pick different hotel to book unclean musty smelling rooms. terrible customer service. the facility has one washer and one dryer located directly in the hallway on the first floor. unfortunately something spilled in my luggage forcing me to wash all of my clothes upon arrival. after running the dry cycle clothes were still wet. brought the issue up to the front desk and requested refund for the laundry machine only to be told refunds could only be given by management. apparently management is only on site from midnight to 7am when went in the morning to request the refund from management was told the front desk personnel was in the bathroom. waited 10 minutes until had to leave for work with no one returning to the desk to assist. no refund no apology no work around and no out of order sign was issued. centrally located easy commute to warner robins afb rooms dirty nearby activities shopping food drinks breakfast provided"}, {"sample_id": "g-ChdDSUhNMG9nS0VJQ0FnTUNRd0tYZi13RRAB", "review_text": "overall score 4 cosas positivas tiene refrigerador hervidor en la pieza gimnasio dispensador de agua en el pasillo opción de limpieza diaria buena ubicación tranquila pero lejana estaciones de metro pieza amplia el primer día reciben con jugo chocolate gratis 14 de febrero. cosas negativas tv sin smart tv ducha incómoda llave de lavamano muy incómoda inodoro sin soporte posterior no entregan frazadas sólo cubrecamas falta de espacio para poner cosas en el baño."}]

    # Convert batch_data to JSON
    # json_data = json.dumps(batch_data['input_columns'], ensure_ascii=False)
    # Convert model_params to JSON
    jsonSchema_str = json.dumps(request.model_params.output_schema or {}, ensure_ascii=False)

    # Construct the prompt
    prompt_template = "{}Follow JSON schema.<JSONSchema>{}</JSONSchema>"
    constructed_prompt = prompt_template.format(json_data, jsonSchema_str)

    response = handler.generate_response(constructed_prompt, labels={"model_id": request.model_params.model_id})
    
    return {"response": response}

# To run the application, use the following command:
# uvicorn your_script_name:app --reload