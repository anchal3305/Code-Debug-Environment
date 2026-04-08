import torch
from bugs import TASKS


class CodeDebugEnv:
    def __init__(self):
        self.tasks = TASKS
        self.current_index = 0
        self.attempts = 0
        self.total_score = 0.0

    def reset(self):
        self.current_index = 0
        self.attempts = 0
        self.total_score = 0.0
        return self._get_observation()

    def step(self, action):
        fixed_code = action.get("fixed_code", "")
        task = self.tasks[self.current_index]

        reward, info = self._run_tests(fixed_code, task)
        self.total_score += reward
        self.attempts += 1
        self.current_index += 1
        done = self.current_index >= len(self.tasks)

        return {
            "state": self._get_observation() if not done else None,
            "reward": round(reward, 2),
            "done": done,
            "info": info
        }

    def state(self):
        return self._get_observation()

    # ── private helpers ──────────────────────────────────────────────

    def _get_observation(self):
        task = self.tasks[self.current_index]
        return {
            "task_id":     task["id"],
            "difficulty":  task["difficulty"],
            "description": task["description"],
            "buggy_code":  task["buggy_code"],
        }

    def _run_tests(self, fixed_code, task):
        passed = 0
        total  = len(task["test_cases"])
        errors = []

        for test in task["test_cases"]:
            try:
                exec_globals = {"torch": torch}
                exec(fixed_code, exec_globals)

                # grab the first function defined in the submitted code
                fn = next(
                    v for v in exec_globals.values() if callable(v)
                )

                # ── special checks ───────────────────────────────────
                if test.get("check") == "normalize":
                    a = torch.tensor(test["a"])
                    result = fn(a)
                    mean_ok = abs(result.mean().item()) < 1e-5
                    std_ok  = abs(result.std().item() - 1.0) < 1e-2
                    if mean_ok and std_ok:
                        passed += 1

                elif test.get("check") == "dim1_softmax":
                    a = torch.tensor(test["a"])
                    result = fn(a)
                    row_sums = result.sum(dim=1)
                    if all(abs(s.item() - 1.0) < 1e-5 for s in row_sums):
                        passed += 1

                # ── standard checks ──────────────────────────────────
                else:
                    # build args dynamically from test keys
                    args = []
                    for key in ["a", "b"]:
                        if key in test:
                            val = test[key]
                            # convert lists → tensors; scalars stay as-is
                            if isinstance(val, list):
                                val = torch.tensor(val, dtype=torch.float32)
                            args.append(val)

                    result   = fn(*args)
                    expected = test["expected"]

                    if isinstance(expected, list):
                        expected = torch.tensor(expected, dtype=torch.float32)
                        if torch.allclose(result.float(), expected, atol=1e-4):
                            passed += 1
                    else:
                        # scalar comparison
                        if abs(float(result) - float(expected)) < 1e-4:
                            passed += 1

            except Exception as e:
                errors.append(str(e))

        reward = passed / total
        return reward, {
            "tests_passed": passed,
            "tests_total":  total,
            "errors":       errors
        }