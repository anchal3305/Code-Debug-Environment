---
title: Code Debugging Environment
emoji: 🐛
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
---

# Code Debugging Environment

An OpenEnv-compatible RL environment where an AI agent fixes buggy PyTorch code.

## Observation Space
| Field | Description |
|---|---|
| task_id | Unique task identifier |
| difficulty | easy / medium / hard |
| description | What the function should do |
| buggy_code | The broken Python function |

## Action Space
`{ "fixed_code": "def solve(a, b):\n    return a + b" }`

## Reward Function
| Outcome | Reward |
|---|---|
| All tests pass | 1.00 |
| Partial pass (k/n) | k/n |
| Syntax error / crash | 0.00 |

## Results
- Agent: Qwen/Qwen2.5-72B-Instruct
- Score: 10/10
- Accuracy: 100%