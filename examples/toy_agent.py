"""A toy 'agent' that writes buggy code, sees the test fail, fixes it and tries again.

No LLM needed. It shows the loop every real coding agent runs, and the trajectory it leaves.

    uv run python examples/toy_agent.py
    uv run gauntlet-sandbox show toy_run.jsonl
"""

from gauntlet_sandbox import Sandbox

TEST = """\
from fizzbuzz import fizzbuzz
assert fizzbuzz(3) == "Fizz"
assert fizzbuzz(5) == "Buzz"
assert fizzbuzz(15) == "FizzBuzz"
assert fizzbuzz(7) == "7"
print("all tests pass")
"""

ATTEMPTS = [
    # first try: classic ordering bug, 15 never reaches FizzBuzz
    """\
def fizzbuzz(n):
    if n % 3 == 0:
        return "Fizz"
    if n % 5 == 0:
        return "Buzz"
    if n % 15 == 0:
        return "FizzBuzz"
    return str(n)
""",
    # second try: fixed
    """\
def fizzbuzz(n):
    if n % 15 == 0:
        return "FizzBuzz"
    if n % 3 == 0:
        return "Fizz"
    if n % 5 == 0:
        return "Buzz"
    return str(n)
""",
]

with Sandbox(trajectory_path="toy_run.jsonl", metadata={"task": "fizzbuzz"}) as sb:
    sb.write_file("test_fizzbuzz.py", TEST)
    for i, code in enumerate(ATTEMPTS, 1):
        sb.write_file("fizzbuzz.py", code)
        result = sb.exec("python test_fizzbuzz.py")
        print(f"attempt {i}: {'PASS' if result.ok else 'FAIL'}")
        if result.ok:
            break

print("trajectory written to toy_run.jsonl")
