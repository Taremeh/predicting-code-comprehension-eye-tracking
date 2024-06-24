quux = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def foo(
    corge: str, waldo: str, grault: Literal["fred", "thud"]
) -> str:
    bongo = quux if grault == "thud" else corge
    eggs = corge if grault == "thud" else quux
    baz = ""
    for garply in waldo:
        if garply.upper() in bongo:
            ham = bongo.find(garply.upper())
            if garply.isupper():
                baz += eggs[ham].upper()
            else:
                baz += eggs[ham].lower()
        else:
            baz += garply
    return baz


def bar(corge: str, waldo: str) -> str:
    return foo(corge, waldo, "fred")