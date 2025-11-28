import sys
sys.path.insert(0, "/data/CosyVoice")

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from final_tts import generate_emotional_tts


app = FastAPI()

class Request(BaseModel):
    text: str
    emotions: list[str]
    percentages: list[str]
    top_k: int = 1


@app.post("/generate")
async def generate(req: Request):

    output = generate_emotional_tts(
        text=req.text,
        emotions=req.emotions,
        percentages=req.percentages,
        top_k=req.top_k
    )

    return {"audio_path": output}


if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=7860)
