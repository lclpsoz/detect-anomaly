import random

class gen_dummy:
    def __init__(self, n: int, min_val: int, max_val: int, min_bias: int, max_bias: int, odd_bias = float):
        self.n = n
        self.min_val = min_val
        self.max_val = max_val
        self.min_bias = min_bias
        self.max_bias = max_bias
        self.odd_bias = odd_bias

    def get_hist_spot(self, bias: bool) -> list:
        ret = [[0 for x in range(256)] for x in range(3)]
        for i in range(self.n):
            for j in range(self.n):
                for channel in range(len(ret)):
                    min_now = self.min_val
                    max_now = self.max_val
                    if(bias and random.random() < self.odd_bias):
                        min_now = self.min_bias
                        max_now = self.max_bias
                    ret[channel][random.randint(min_now, max_now)]+=1 

        return ret