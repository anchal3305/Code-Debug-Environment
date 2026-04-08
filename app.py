import gradio as gr
from huggingface_hub import InferenceClient
from environment import CodeDebugEnv

client = InferenceClient()
env = CodeDebugEnv()

def run_agent():
    obs = env.reset()
    output = "Code Debugging Environment — Agent Run\n"
    output += "=" * 50 + "\n"

    while True:
        task_id    = obs["task_id"]
        difficulty = obs["difficulty"]
        description = obs["description"]
        buggy_code = obs["buggy_code"]

        # call LLM to fix the code
        response = client.chat_completion(
            model="Qwen/Qwen2.5-72B-Instruct",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert Python and PyTorch debugger. "
                        "Return ONLY the fixed Python function. "
                        "No explanation, no markdown, no comments. "
                        "Keep the function name as 'solve'. "
                        "torch is already imported."
                    )
                },
                {
                    "role": "user",
                    "content": f"Task: {description}\n\nBuggy code:\n{buggy_code}"
                }
            ],
            max_tokens=200,
        )
        fixed_code = response.choices[0].message.content.strip()

        result = env.step({"fixed_code": fixed_code})
        reward = result["reward"]
        info   = result["info"]

        output += f"\n🔍 Task: {task_id} | Difficulty: {difficulty.upper()}\n"
        output += f"   {description}\n"
        output += f"\n   Buggy code:\n   {buggy_code}\n"
        output += f"\n   Fixed code:\n   {fixed_code}\n"
        output += f"\n   ✅ Tests passed: {info['tests_passed']}/{info['tests_total']}\n"
        output += f"   🏆 Reward: {reward:.2f}\n"
        output += "-" * 50 + "\n"

        if result["done"]:
            total = sum(
                result["reward"] if result["done"] else 0
                for result in [result]
            )
            break
        obs = result["state"]

    output += f"\n🎯 Run complete! Check results above."
    return output

demo = gr.Interface(
    fn=run_agent,
    inputs=[],
    outputs=gr.Textbox(label="Agent Results", lines=40),
    title="🐛 Code Debugging Environment",
    description="Click Run to watch an LLM agent fix buggy PyTorch code in real time!",
)

demo.launch()