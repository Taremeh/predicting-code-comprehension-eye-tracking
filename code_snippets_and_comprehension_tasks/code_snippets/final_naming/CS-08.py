def resolve(source, filename):
    if _looks_like_filename(source):
        return _resolve_filename(source, filename)

    if isinstance(source, str):
        source = source.splitlines()

    # At this point "source" is not a str.
    if not filename:
        filename = None
    elif not isinstance(filename, str):
        raise TypeError(f'filename should be str (or None), got {filename!r}')
    else:
        filename, _ = _resolve_filename(filename)
    return source, filename


def _looks_like_filename(value):
    if not isinstance(value, str):
        return False
    return value.endswith(('.c', '.h'))


def _resolve_filename(filename, alt=None):
    if os.path.isabs(filename):
        ...
#        raise NotImplementedError
    else:
        filename = os.path.join('.', filename)

    if not alt:
        alt = filename
    elif os.path.abspath(filename) == os.path.abspath(alt):
        alt = filename
    else:
        raise ValueError(f'mismatch: {filename} != {alt}')
    return filename, alt