def foo(qux):
    return bar(qux)


def bar(qux):
    corge = qux.split('=')

    if len(corge) != 2:
        raise argparse.ArgumentTypeError('"%s" must be in the format "key=value"' % qux)

    return corge[0], corge[1]