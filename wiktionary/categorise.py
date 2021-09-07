import unidecode


def categorise(source, target):
    """
    Return a numerical label indicating inflection class.

    Args:
        source (str): source with gender as first character ("<f> k a t z e ")
        target (str): (predicted) target

    Returns:
        int indicating class, ranges from 0 - 6
    """
    assert "<" in source.split()[0], "Your source sequence has no gender tag!"

    source = source.replace(" </s>", "")
    target = target.replace(" </s>", "")
    source = unidecode.unidecode(source).split()[1:]
    target = unidecode.unidecode(target).split()

    # zero or epsilon
    if target == source:
        category = 2
    elif len(target) > len(source) and target[:len(source)] == source:
        # (e)n
        if target[-1:] == ["n"]:
            category = 0
        # e
        elif target[-1:] == ["e"]:
            category = 1
        # e r
        elif target[-2:] == ["e", "r"]:
            category = 3
        # s
        elif target[-1:] == ["s"]:
            category = 4
        # repeated input but odd suffix
        elif target[:len(source)] == source:
            category = 5
    else:
        # didn't even repeat the input
        category = 6
    return category
