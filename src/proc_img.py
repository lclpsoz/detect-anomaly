import random
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import zipfile
from tqdm import tqdm

class procImg:
    """Class to generate and processes images that will be used as training data.
    
    Attributes
    ----------
        n : int
            Dimension of the image (n x n).
        min_val : int
            Minimum intensity for a pixel without bias.
        max_val : int
            Maximum intensity for a pixel without bias.
        min_bias : int
            Minimum intensity for a pixel with bias.
        max_bias : int
            Maximum intensity for a pixel with bias.
        odd_bias : int
            Percentual chance of a pixel, in a image with anomaly, being applied
                bias.
        skin_color : list
            List of 3 uint of 8 bits, representing the skin color of the image,
            as RGB.
        spot_border_color : float
            Color of the border of the spot.
        spot_radius : float
            The radius of the spot.
        channels_bias : list
            List of booleans to indiquate if the RGB channel will have bias.
    """
    def __init__(self, n: int, min_val: int, max_val: int, min_bias: int,
                    max_bias: int, odd_bias: float, skin_color: list,
                    spot_border_color:float, spot_radius: float,
                    channels_bias: list):
        self.n = n
        self.min_val = min_val
        self.max_val = max_val
        self.min_bias = min_bias
        self.max_bias = max_bias
        self.odd_bias = odd_bias
        self.skin_color = skin_color
        self.spot_border_color = spot_border_color
        self.spot_radius = spot_radius
        self.channels_bias = channels_bias

    @staticmethod
    def imshow (img):
        if (hasattr(img[0][0], '__len__')):
            plt.imshow (img, cmap='gray', interpolation='nearest')
        else:
            plt.imshow (img, interpolation='nearest')
        plt.show()

    def get_vals_spot(self, img:list):
        ret = [[] for x in range(3)]
        for i in range(self.n):
            for j in range(self.n):
                if(not 0 in img[i][j] and not 255 in img[i][j]):
                    for channel in range(len(ret)):
                        ret[channel].append(img[i][j][channel])

        return ret


    def get_ga_list_spot(self, img: list) -> list:
        """Generate a histogram from a nparray uint8.
        Args:
            img (nparray uint8): Image to generate histogram.
        Returns:
            list: Matrix 3x256 with histogram of the img.
        """
        ret = [[0 for x in range(256)] for x in range(3)]
        for i in range(self.n):
            for j in range(self.n):
                if(not 0 in img[i][j] and not 255 in img[i][j]):
                    for channel in range(len(ret)):
                        ret[channel][img[i][j][channel]]+=1 

        return ret

    def generate_img(self, apply_bias: bool):
        img = np.array([[self.skin_color for x in range(self.n)] for x in range(self.n)], dtype=np.uint8)
        md = [self.n//2 + random.randint(-8, 8), self.n//2 + random.randint(-8, 8)]
        for i in range(self.n):
            for j in range(self.n):
                dist = int(np.linalg.norm(md-np.array([i, j])))
                if(dist < self.spot_radius+1):
                    if(dist > self.spot_radius-2):
                        img[i][j] = self.spot_border_color
                    else:
                        for channel in range(3):
                            min_now = self.min_val
                            max_now = self.max_val
                            if(apply_bias and self.channels_bias[channel] and random.random() < self.odd_bias):
                                min_now = self.min_bias
                                max_now = self.max_bias
                            img[i][j][channel] = random.randint(min_now, max_now)

        return img

    def generate_img_batch(self, folder_name: str, qnt: int, apply_bias: bool):
        img_path = os.path.join('..', 'img')
        if(not os.path.exists(img_path)):
            os.makedirs(img_path)
        if(not os.path.exists(os.path.join(img_path, folder_name))):
            os.makedirs(os.path.join(img_path, folder_name))

        if (apply_bias):
            pref = "sick_"
        else:
            pref = "health_"
        for i in tqdm(range(qnt)):
            with open(os.path.join(img_path, folder_name, pref + str(i) + '.json'), 'w') as fl:
                json.dump(self.generate_img(apply_bias).tolist(), fl)

    def generate_dataset(self, zip_name : str) -> None:
        img_path = os.path.join('..', 'img')
        if(not os.path.exists(img_path)):
            return None
        dataset = []
        with zipfile.ZipFile(os.path.join(img_path, zip_name + '.zip'), 'r') as z:
            lst = z.namelist()
            for i in tqdm(range(len(lst))):
                filename = lst[i]
                if(filename != "img_batch_n_100_r_35_p_0-01_train/"):
                    with z.open(filename) as fl:
                        img = np.array(json.loads(fl.read().decode('utf-8')))
                        dataset.append([self.get_ga_list_spot(img), 'sick' in filename])
        with open(os.path.join(img_path, "ga_list_" + zip_name + '.json'), 'w') as fl:
            json.dump(dataset, fl)

    @staticmethod
    def get_dataset(id : str) -> list:
        """Loads requisited dataset from json file."""
        fl_path = os.path.join('..', 'img', id + '.json')
        if(os.path.exists(fl_path)):
            with open(fl_path, 'r') as fl:
                return json.load(fl)
        else:
            return None