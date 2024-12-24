import asyncio
import json
import os
import aiofiles
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

# 设置API参数
base_url = "http://180.184.175.69:3000/v1"
api_key = "sk-x23IMWeR094xum1R4QkUPhSq9pP1XFOjDWrBGldtUVbAjWQJ"
client = AsyncOpenAI(base_url=base_url, api_key=api_key)

# 并发限制
MAX_CONCURRENT = 150
model = "gpt-4o-2024-11-20"

# 重试机制
def retry_error_callback(retry_state):
    exception = retry_state.outcome.exception()
    print(f"Retry attempt {retry_state.attempt_number} failed: {type(exception).__name__} - {str(exception)}")
    return None

# 保存中间结果
async def save_temp_results(results, save_path, current_count):
    temp_file = f"{save_path}_temp_result.json"
    last_file = f"{save_path}_temp_result_last.json"

    if os.path.exists(temp_file):
        try:
            os.replace(temp_file, last_file)
        except Exception as e:
            print(f"Failed to rename old temp file: {e}")

    async with aiofiles.open(temp_file, 'w', encoding='utf-8') as f:
        await f.write(json.dumps(results, ensure_ascii=False, indent=2))

# 调用模型接口
@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=1, max=15), retry_error_callback=retry_error_callback)
async def get_chat_completion(message: dict, semaphore, retry_count=0) -> dict:
    # prompt = f"Question: {message['data']['question']}\nOptions: {', '.join(message['data']['options'])}"
    # print(prompt)
    # prompt = prompt.replace('\n', '').replace(' ', '') 
    try:
        async with semaphore:
            # 生成 Prompt
            prompt_q = f"Question: {message['data']['question']}\n"
            prompt_o = f"Options: {', '.join(message['data']['options'])}"
            prompt_o = prompt_o.replace('\n', '')
            prompt = prompt_q + prompt_o
            message["prompt"] = prompt  # 将生成的 Prompt 存入结果
            
            # 调用模型
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Answer the question correctly based on the given options. Directly output the answer without any other words."},
                    {"role": "user", "content": prompt}],
            )
            response_result = response.choices[0].message.content.strip()
            
            # 保存模型返回结果
            message["generated_answer"] = response_result
            return message
    except Exception as e:
        print(f"Error in get_chat_completion for message  {type(e).__name__} - {str(e)}")
        raise

# 模型测试和结果保存
async def request_model(prompts, save_path="results"):
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    results = []
    correct_count = 0
    completed_count = 0

    async def wrapped_get_chat_completion(prompt, index):
        nonlocal completed_count, correct_count
        try:
            result = await get_chat_completion(prompt, semaphore)
            completed_count += 1

            # 比较答案正确性
            correct_answer = prompt["data"]["answer"].strip().lower()
            if result["generated_answer"].strip().lower() == correct_answer:
                result["is_correct"] = True
                correct_count += 1
            else:
                result["is_correct"] = False

            # 每完成100条保存一次
            if completed_count % 1000 == 0:
                await save_temp_results(results, save_path, completed_count)

            return index, result
        except Exception as e:
            print(f"Task failed after all retries with error: {e}")
            return index, None

    # 创建异步任务
    tasks = [wrapped_get_chat_completion(prompt, i) for i, prompt in enumerate(prompts)]
    results = [None] * len(prompts)

    for future in tqdm.as_completed(tasks, total=len(tasks), desc="Processing prompts"):
        index, result = await future
        results[index] = result

    # 保存最终结果
    await save_temp_results(results, save_path, "final")

    total_count = len(prompts)
    accuracy = correct_count / total_count * 100 if total_count > 0 else 0
    print(f"Total questions: {total_count}")
    print(f"Correct answers: {correct_count}")
    print(f"Accuracy: {accuracy:.2f}%")

    return results

if __name__ == "__main__":
    file_path = "bio.jsonl"
    with open(file_path, 'r', encoding="utf-8") as f:
        prompts = [json.loads(line) for line in f]

    # 截取部分测试数据
    # prompts = prompts[0:5]

    results = asyncio.run(request_model(prompts))

    save_path = file_path.replace(".json", "") + f"_{model}_result.json"
    with open(save_path, 'w', encoding="utf-8") as f:
        f.write(json.dumps(results, ensure_ascii=False, indent=2))
