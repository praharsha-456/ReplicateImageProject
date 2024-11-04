from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
print(REPLICATE_API_TOKEN)
REPLICATE_API_URL = "https://api.replicate.com/v1/predictions"

# Initialize FastAPI app
app = FastAPI()

# Define the input model for generating images
class ImageGenerationRequest(BaseModel):
    prompt: str
    model_version: str = "stability-ai/stable-diffusion:ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4"  # Replace with actual model version ID

# Define the response model for better structure
class ImageGenerationResponse(BaseModel):
    image_url: str

@app.post("/generate_image", response_model=ImageGenerationResponse)
async def generate_image(request: ImageGenerationRequest):
    headers = {
        "Authorization": f"Token {REPLICATE_API_TOKEN}",
        "Content-Type": "application/json",
    }
    print(REPLICATE_API_TOKEN)
    payload = {
        "version": request.model_version,
        "input": {
            "prompt": request.prompt
        }
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(REPLICATE_API_URL, json=payload, headers=headers)
        print(response.text)
        # Handle errors from Replicate API
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Error generating image.")

        result = response.json()
        output_url = result.get("output", [None])[0]

        if not output_url:
            raise HTTPException(status_code=500, detail="Failed to retrieve image URL from Replicate.")

        return ImageGenerationResponse(image_url=output_url)
