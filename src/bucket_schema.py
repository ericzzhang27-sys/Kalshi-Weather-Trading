from dataclasses import dataclass,field
import numpy as np
@dataclass
class Bucket:
    location: str
    left_bound: int | None = None
    right_bound: int | None = None
    inner_bounds: list[float] = field(init = False)
    num_inner_bounds: int = 2
    def __post_init__(self):
        if self.left_bound is None and self.right_bound is None:
            raise ValueError("left and right bound cannot both be none")
        if self.right_bound is not None and self.left_bound is not None and self.right_bound <= self.left_bound:
            raise ValueError("right bound must be greater than left bound")
        if not (isinstance(self.left_bound, int) or self.left_bound is None) or not (isinstance(self.right_bound, int) or self.right_bound is None):
            raise TypeError("Bounds must be int or None")
        if self.num_inner_bounds < 1:
            raise ValueError("num_inner_bounds must be greater than 0")
        if self.left_bound is None:
            self.left_bound = -np.inf
        if self.right_bound is None:
            self.right_bound = np.inf
        self.inner_bounds = self.calculate_thresholds(self.num_inner_bounds)
    def calculate_thresholds(self, num_boundaries: int) -> list[float]:
        left = float(self.left_bound if self.left_bound is not None else -np.inf)
        right = float(self.right_bound if self.right_bound is not None else np.inf)

        if self.left_bound is None:
            return [left, right - 0.5]
        if self.right_bound is None:
            return [left - 0.5, right]

        return [left - 0.5, right + 0.5]

