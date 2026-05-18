from dataclasses import dataclass,field

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
            self.left_bound = 0
        if self.right_bound is None:
            self.right_bound = 999
        self.inner_bounds = self.calculate_thresholds(self.num_inner_bounds)
    def calculate_thresholds(self, num_boundaries: int) -> list[float]:
        boundaries: float = (self.right_bound-self.left_bound)/num_boundaries
        res=[]
        for i in range(num_boundaries+1):
            res.append(self.left_bound+boundaries*i)
        return res

