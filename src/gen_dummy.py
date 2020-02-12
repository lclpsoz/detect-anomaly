import random
import os
import json

# Dummy class to be used by GA until
# definitive version is ready.
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

    def __save_dataset(self, dataset : list, id : str) -> None:
        img_path = os.path.join('..', 'img')
        if(not os.path.exists(img_path)):
            os.makedirs(img_path)
        with open(os.path.join(img_path, id + ".json"), 'w') as fl:
            json.dump(dataset, fl)

    def generate_dataset(self, dataset_size : int, anomaly_proportion : float, id : str) -> None:
        dataset = []
        amount_anomaly = dataset_size * anomaly_proportion
        for i in range(dataset_size):
            if(i%100 == 0):
                print(i)
            if(i < amount_anomaly):
                dataset.append((self.get_hist_spot(True), True))
            else:
                dataset.append((self.get_hist_spot(False), False))

        self.__save_dataset(dataset, id)

    @staticmethod
    def get_dataset(id : str) -> list:
        fl_path = os.path.join('..', 'img', id + '.json')
        if(os.path.exists(fl_path)):
            with open(fl_path, 'r') as fl:
                return json.load(fl)
        else:
            return None