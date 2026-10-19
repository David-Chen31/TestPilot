"""沙箱测试用的小型“被测仓库”。"""


def add(a: int, b: int) -> int:
    return a + b


def clamp(x: int, low: int, high: int) -> int:
    if low > high:
        raise ValueError("low must not exceed high")
    if x < low:
        return low
    if x > high:
        return high
    return x
