import torch

TASKS = [
    {
        "id": "bug_001",
        "difficulty": "easy",
        "description": "Fix the function so it correctly adds two tensors.",
        "buggy_code": "def solve(a, b):\n    return a - b",
        "test_cases": [
            {"a": [1, 2], "b": [3, 4], "expected": [4, 6]},
            {"a": [0, 5], "b": [5, 5], "expected": [5, 10]},
            {"a": [10, 20], "b": [1, 2], "expected": [11, 22]},
        ]
    },
    {
        "id": "bug_002",
        "difficulty": "easy",
        "description": "Fix the function so it multiplies each element by 2.",
        "buggy_code": "def solve(a):\n    return a + 2",
        "test_cases": [
            {"a": [1, 2, 3], "expected": [2, 4, 6]},
            {"a": [5, 10], "expected": [10, 20]},
            {"a": [0, 1], "expected": [0, 2]},
        ]
    },
    {
        "id": "bug_003",
        "difficulty": "easy",
        "description": "Fix the function so it returns the max value in a tensor.",
        "buggy_code": "def solve(a):\n    return torch.min(a)",
        "test_cases": [
            {"a": [1, 5, 3], "expected": 5},
            {"a": [10, 2, 8], "expected": 10},
            {"a": [0, 0, 1], "expected": 1},
        ]
    },
    {
        "id": "bug_004",
        "difficulty": "medium",
        "description": "Fix the matrix multiplication — wrong operator used.",
        "buggy_code": "def solve(a, b):\n    return a * b",
        "test_cases": [
            {"a": [[1, 2], [3, 4]], "b": [[1, 0], [0, 1]], "expected": [[1, 2], [3, 4]]},
            {"a": [[1, 0], [0, 1]], "b": [[5, 6], [7, 8]], "expected": [[5, 6], [7, 8]]},
        ]
    },
    {
        "id": "bug_005",
        "difficulty": "medium",
        "description": "Fix the function so it returns the sum along dimension 1.",
        "buggy_code": "def solve(a):\n    return torch.sum(a, dim=0)",
        "test_cases": [
            {"a": [[1, 2], [3, 4]], "expected": [3, 7]},
            {"a": [[5, 5], [1, 1]], "expected": [10, 2]},
        ]
    },
    {
        "id": "bug_006",
        "difficulty": "medium",
        "description": "Fix the function to apply softmax along the correct dimension (dim=1).",
        "buggy_code": "def solve(a):\n    return torch.softmax(a, dim=0)",
        "test_cases": [
            {"a": [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], "check": "dim1_softmax"},
        ]
    },
    {
        "id": "bug_007",
        "difficulty": "medium",
        "description": "Fix the function so it flattens the tensor correctly.",
        "buggy_code": "def solve(a):\n    return a.reshape(1, -1)",
        "test_cases": [
            {"a": [[1, 2], [3, 4]], "expected": [1, 2, 3, 4]},
            {"a": [[5, 6], [7, 8]], "expected": [5, 6, 7, 8]},
        ]
    },
    {
        "id": "bug_008",
        "difficulty": "hard",
        "description": "Fix the function so it normalizes a tensor to have mean=0 and std=1.",
        "buggy_code": "def solve(a):\n    return (a - torch.mean(a)) / torch.mean(a)",
        "test_cases": [
            {"a": [1.0, 2.0, 3.0, 4.0], "check": "normalize"},
        ]
    },
    {
        "id": "bug_009",
        "difficulty": "hard",
        "description": "Fix the function to compute element-wise mean of two tensors.",
        "buggy_code": "def solve(a, b):\n    return (a + b) / 4",
        "test_cases": [
            {"a": [2.0, 4.0], "b": [4.0, 8.0], "expected": [3.0, 6.0]},
            {"a": [0.0, 0.0], "b": [2.0, 4.0], "expected": [1.0, 2.0]},
        ]
    },
    {
        "id": "bug_010",
        "difficulty": "hard",
        "description": "Fix the function to return the dot product of two 1D tensors.",
        "buggy_code": "def solve(a, b):\n    return torch.cross(a, b)",
        "test_cases": [
            {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "expected": 32.0},
            {"a": [1.0, 0.0, 0.0], "b": [1.0, 0.0, 0.0], "expected": 1.0},
        ]
    },
]