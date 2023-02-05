import numpy as np
from carlo import base_sampler


class WangLandau(base_sampler.BaseSampler):
    def __init__(self) -> None:
        super().__init__()
