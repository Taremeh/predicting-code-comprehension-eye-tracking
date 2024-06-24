def peek_and_iter(items):
    if not items:
        return None, None
    items = iter(items)
    try:
        peeked = next(items)
    except StopIteration:
        return None, None
    def chain():
        yield peeked
        yield from items
    return chain(), peeked