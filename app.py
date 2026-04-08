import gradio as gr
from environment import CodeDebugEnv

env = CodeDebugEnv()

def run_demo(buggy_code, description):
    obs = env.reset()
    return f"Task: {obs['task_id']}\nDifficulty: {obs['difficulty']}\nDescription: {obs['description']}\nBuggy Code:\n{obs['buggy_code']}"

demo = gr.Interface(
    fn=run_demo,
    inputs=[
        gr.Textbox(label="Buggy Code"),
        gr.Textbox(label="Description")
    ],
    outputs=gr.Textbox(label="Environment Output"),
    title="Code Debugging Environment",
    description="An OpenEnv RL environment where an AI agent fixes buggy PyTorch code."
)

demo.launch()