import asyncio
import json
import os
import aiofiles
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

# old
base_url="http://180.184.174.65:3000/v1"
api_key="sk-x23IMWeR094xum1R4QkUPhSq9pP1XFOjDWrBGldtUVbAjWQJ"
client = AsyncOpenAI(base_url=base_url, api_key=api_key)

# 设置并发限制
MAX_CONCURRENT = 150

model = "gemini-1.5-pro-002"

def retry_error_callback(retry_state):
    exception = retry_state.outcome.exception()
    print(f"Retry attempt {retry_state.attempt_number} failed: {type(exception).__name__} - {str(exception)}")
    return None

async def save_temp_results(results, save_path, current_count):
    temp_file = f"{save_path}_temp_result.json"
    last_file = f"{save_path}_temp_result_last.json"
    
    # 如果存在旧文件,则重命名为last
    if os.path.exists(temp_file):
        try:
            os.replace(temp_file, last_file)
        except Exception as e:
            print(f"Failed to rename old temp file: {e}")
    
    # 保存新的临时文件
    async with aiofiles.open(temp_file, 'w', encoding='utf-8') as f:
        await f.write(json.dumps(results, ensure_ascii=False, indent=2))
        
@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=1, max=15), retry_error_callback=retry_error_callback)
async def get_chat_completion(message: str, semaphore, retry_count=0) -> str:
    try:
        async with semaphore:  # 使用传入的信号量限制并发
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "hello"},
                    {"role": "user", "content": "hello"}],
                # timeout=80
            )
            response_result = response.choices[0].message.content
            message[model] = response_result
            return message
    except Exception as e:
        print(f"Error in get_chat_completion for message  {type(e).__name__} - {str(e)}")
        raise

async def request_model(prompts, save_path="results"):
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    results = []
    completed_count = 0
    
    async def wrapped_get_chat_completion(prompt, index):
        nonlocal completed_count
        try:
            result = await get_chat_completion(prompt, semaphore)
            completed_count += 1
            
            # 每完成100条保存一次
            if completed_count % 1000 == 0:
                await save_temp_results(results, save_path, completed_count)
                
            return index, result
        except Exception as e:
            print(f"Task failed after all retries with error: {e}")
            return index, None

    tasks = [wrapped_get_chat_completion(prompt, i) for i, prompt in enumerate(prompts)]
    
    # 使用固定大小的列表预分配空间
    results = [None] * len(prompts)
    
    for future in tqdm.as_completed(tasks, total=len(tasks), desc="Processing prompts"):
        index, result = await future
        results[index] = result

    # 最后保存一次完整结果
    await save_temp_results(results, save_path, "final")
    
    return results

if __name__ == "__main__":
    file_path = "bio.jsonl"
    with open(file_path, 'r', encoding="utf-8") as f:
        prompts = json.load(f)

    # prompts = prompts[0:5]

    results = asyncio.run(request_model(prompts))

    save_path = file_path.replace(".json", "") + f"_{model}_result.json"
    with open(save_path, 'w', encoding="utf-8") as f:
        f.write(json.dumps(results))