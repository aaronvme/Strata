def permutation(n: Int, seed: Int = 42) -> List[Int]:
    """Generates a pseudo-random permutation of range(0, n) using Fisher-Yates shuffle."""
    var indices = List[Int](capacity=n)
    for i in range(n):
        indices.append(i)

    if n <= 1:
        return indices^

    var state = seed
    if state == 0:
        state = 42

    for i in range(n - 1, 0, -1):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        var j = state % (i + 1)
        var temp = indices[i]
        indices[i] = indices[j]
        indices[j] = temp

    return indices^

def shuffle[T: Copyable & Movable](mut list: List[T], seed: Int = 42):
    """In-place Fisher-Yates shuffle on a List."""
    var n = len(list)
    if n <= 1:
        return

    var state = seed
    if state == 0:
        state = 42

    for i in range(n - 1, 0, -1):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        var j = state % (i + 1)
        var temp = list[i]
        list[i] = list[j]
        list[j] = temp
