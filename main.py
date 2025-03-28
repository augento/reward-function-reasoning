from typing import Any, Dict, List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from openai.types.chat import ChatCompletionMessageParam
import re


app = FastAPI()

class Completion(BaseModel):
    prompt_messages: List[ChatCompletionMessageParam]
    completion: str
    extra_data: Optional[Dict[str, Any]] = None

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses] 
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses] 
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


async def grading_function(completion: Completion) -> float:
    prompts = completion.prompt_messages
    completions = [[{"content": completion.completion}]]
    answer = [completion.extra_data.get("answer", "")] if completion.extra_data else [""]
    
    correctness_rewards = correctness_reward_func(prompts, completions, answer)
    int_rewards = int_reward_func(completions)
    strict_format_rewards = strict_format_reward_func(completions)
    soft_format_rewards = soft_format_reward_func(completions)
    xmlcount_rewards = xmlcount_reward_func(completions)
    
    total_reward = sum([
        correctness_rewards[0],
        int_rewards[0],
        strict_format_rewards[0],
        soft_format_rewards[0],
        xmlcount_rewards[0]
    ])
    
    return total_reward

@app.post("/grade")
async def grade(completion: Completion):
    reward = await grading_function(completion)
    return {"reward": reward}
