def foo(n: int) -> int:
    if n < 0:
        raise ValueError("Negative arguments are not supported")
    return bar(n)[0]

def bar(n: int) -> tuple[int, int]:
    if n == 0:
        return (0, 1)

    a, b = bar(n // 2)
    c = a * (b * 2 - a)
    d = a * a + b * b
    return (d, c + d) if n % 2 else (c, d)