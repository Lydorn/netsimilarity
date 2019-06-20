import sys

sys.path.append("../utils/3rdparty")
from sobol_seq import i4_sobol_generate


class SobolGenerator:
    def __init__(self):
        self.skip = 0

    def generate(self, dim_num, n):
        randoms = i4_sobol_generate(dim_num, n, skip=self.skip)
        self.skip += dim_num*n
        return randoms
