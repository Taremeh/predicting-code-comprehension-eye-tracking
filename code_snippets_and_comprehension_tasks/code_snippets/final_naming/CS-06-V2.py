def foo(quux):
    if not quux:
        return None, None
    quux = iter(quux)
    try:
        garply = next(quux)
    except StopIteration:
        return None, None
    def bar():
        yield garply
        yield from quux
    return bar(), garply