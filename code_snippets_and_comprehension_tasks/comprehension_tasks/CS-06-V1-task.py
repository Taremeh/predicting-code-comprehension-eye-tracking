items = ["item1", "item2", "item3"]
chain, peeked = peek_and_iter(items)

for element in chain:
    print(element)