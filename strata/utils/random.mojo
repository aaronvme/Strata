struct PRNG(Copyable, Movable):
    """64-bit SplitMix64 pseudo-random number generator with unbiased range generation.
    """

    var state: UInt64

    def __init__(out self, seed: Int = 42):
        var s = UInt64(Int64(seed))
        if s == 0:
            s = 42
        self.state = s

    def next_u64(mut self) -> UInt64:
        self.state = (self.state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        var z = self.state
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB
        return z ^ (z >> 31)

    def next_int(mut self, upper_bound: Int) -> Int:
        """Returns a pseudo-random integer in [0, upper_bound)."""
        if upper_bound <= 1:
            return 0
        var bound = UInt64(upper_bound)
        var threshold = (0xFFFFFFFFFFFFFFFF - bound + 1) % bound
        while True:
            var r = self.next_u64()
            if r >= threshold:
                return Int(r % bound)


def permutation(n: Int, seed: Int = 42) -> List[Int]:
    """Generates a pseudo-random permutation of range(0, n) using Fisher-Yates shuffle.
    """
    var indices = List[Int](capacity=n)
    for i in range(n):
        indices.append(i)

    if n <= 1:
        return indices^

    var rng = PRNG(seed)
    for i in range(n - 1, 0, -1):
        var j = rng.next_int(i + 1)
        var temp = indices[i]
        indices[i] = indices[j]
        indices[j] = temp

    return indices^


def shuffle[T: Deinitable & Copyable](mut list: List[T], seed: Int = 42):
    """In-place Fisher-Yates shuffle on a List."""
    var n = len(list)
    if n <= 1:
        return

    var rng = PRNG(seed)
    for i in range(n - 1, 0, -1):
        var j = rng.next_int(i + 1)
        var temp = list[i].copy()
        list[i] = list[j].copy()
        list[j] = temp^
