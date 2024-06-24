def foo(a: int, b: int) -> float:
    return a * foo(a, (b - 1)) if b else 1