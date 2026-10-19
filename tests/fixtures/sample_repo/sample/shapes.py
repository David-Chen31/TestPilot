from sample.calc import clamp


class Rect:
    def __init__(self, w: float, h: float):
        self.w = w
        self.h = h

    @property
    def area(self) -> float:
        return self.w * self.h

    def scale(self, k: float) -> "Rect":
        return Rect(self.w * k, self.h * k)

    def clamp_width(self, low: int, high: int) -> "Rect":
        return Rect(clamp(self.w, low, high), self.h)
