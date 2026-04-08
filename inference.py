from huggingface_hub import InferenceClient
from environment import CodeDebugEnv

# ── setup ──────────────────────────────────────────────────────────
client = InferenceClient()
env    = CodeDebugEnv()

# ── agent: calls Mistral to fix buggy code ─────────────────────────
def fix_code(buggy_code, description):
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
    return response.choices[0].message.content.strip()


# ── main loop ──────────────────────────────────────────────────────
def run():
    obs = env.reset()
    total_reward = 0.0
    total_tasks  = len(env.tasks)

    print("=" * 50)
    print("   Code Debugging Environment — Agent Run")
    print("=" * 50)

    while True:
        task_id     = obs["task_id"]
        difficulty  = obs["difficulty"]
        buggy_code  = obs["buggy_code"]
        description = obs["description"]

        print(f"\n🔍 Task: {task_id} | Difficulty: {difficulty.upper()}")
        print(f"   {description}")
        print(f"\n   Buggy code:\n   {buggy_code}")

        # agent proposes a fix
        fixed_code = fix_code(buggy_code, description)
        print(f"\n   Fixed code:\n   {fixed_code}")

        # environment evaluates the fix
        result = env.step({"fixed_code": fixed_code})
        reward = result["reward"]
        info   = result["info"]
        total_reward += reward

        print(f"\n   ✅ Tests passed: {info['tests_passed']}/{info['tests_total']}")
        print(f"   🏆 Reward: {reward:.2f}")

        if info["errors"]:
            print(f"   ⚠️  Errors: {info['errors'][0]}")

        if result["done"]:
            break

        obs = result["state"]

    # ── final summary ──────────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"   Final Score: {total_reward:.2f} / {total_tasks:.1f}")
    print(f"   Accuracy:    {(total_reward / total_tasks) * 100:.1f}%")
    print("=" * 50)


if __name__ == "__main__":
    run()