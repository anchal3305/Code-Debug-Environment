# Code Debugging Environment

An OpenEnv-compatible RL environment where an AI agent fixes buggy PyTorch code.

## What it does
The agent receives broken Python/PyTorch functions and must return corrected versions.
The environment runs fixes against a hidden test suite and returns a reward.

## Observation Space
| Field | Description |
|---|---|
| task_id | Unique task identifier |
| difficulty | easy / medium / hard |
| description | What the function should do |
| buggy_code | The broken Python function |

## Action Space
```json
{ "fixed_code": "def solve(a, b):\n    return a + b" }
```

## Reward Function
| Outcome | Reward |
|---|---|
| All tests pass | 1.00 |
| Partial pass (k/n) | k/n |
| Syntax error / crash | 0.00 |

## How to Run
```bash
pip install -r requirements.txt
python inference.py
```

## Results
- Agent: Qwen/Qwen2.5-72B-Instruct
- Score: 10/10
- Accuracy: 100%