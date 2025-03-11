from typing import Any, Dict, List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from openai.types.chat import ChatCompletionMessageParam

app = FastAPI()

class Completion(BaseModel):
    prompt_messages: List[ChatCompletionMessageParam]
    completion: str
    extra_data: Optional[Dict[str, Any]] = None


async def grading_function(completion: Completion) -> float:
    # Your grading function here
    return 0.5

@app.post("/grade")
async def grade(completion: Completion):
    reward = await grading_function(completion)
    return {"reward": reward}
