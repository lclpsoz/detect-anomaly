import numpy as np
import cv2 as cv
import random
import os
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import zipfile
from tqdm import tqdm

class procImg:
    
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
    def find_countours_img(img: list) -> list:
        
        kernel = np.ones((3, 3, np.uint8))

        if len(img) == 3:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        blur = cv.GaussianBlur(gray, (5, 5), 0)
        ret, thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        
        dilation = cv.dilate(cv.bitwise_not(thresh),kernel, iterations=1)
        
        contours, hierarchy = cv.findContours(dilation, cv.RETR_TREE,cv.CHAIN_APPROX_TC89_KCOS)
        
        return contours 
    
    @staticmethod
    def get_biggest_lesion(contours):
        
        cnt = []
        contour_area = 0
        
        for x in range(len(contours)):
            
            if cv.contourArea(contours[x]) > contour_area:
                contour_area = cv.contourArea(contours[x])
                cnt = contours[x]
        if len(cnt.shape) == 3:
            cnt = [cnt]
        
        return cnt
    
    @staticmethod
    def rect_img(cnt, img: list) -> tuple:
        
        left = img.shape[1]
        top = img.shape[0]
        right = 0
        bottom = 0
        
        for i in range(len(cnt)):
            for j in range(len(cnt[i])):
                
                x, y = cnt[i][j][0]
                
                if x < left:
                    left = x
                if x > right:
                    right = x
                if y < top:
                    top = y
                if y > bottom:
                    bottom = y
        return (left, top, right, bottom)
    
    @staticmethod
    def grabcut_img(img, rect, cnt, rgb = (0, 255, 0), thicc = 1) -> list:
        
        cv.drawContours(img, cnt, -1, rgb, thicc)
        
        mask = np.zeros(img.shape[:2], np.uint8)

        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        
        cv.grabCut(img,mask,rect,bgdModel,fgdModel,5, cv.GC_INIT_WITH_RECT)
        
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        
        return img*mask2[:,:,np.newaxis]
    
    @staticmethod
    def imshow (self, img):
        #Verify if the img is RGB and converts it
        if (hasattr(img[0][0], '__len__')):
            plt.imshow (img, cmap='gray', interpolation= 'nearest')
        else:
            plt.imshow (img, interpolation='nearest')
        plt.show()
    
    def get_vals_spot(self, img:list):
        
        ret = [[]for x in range(3)]

        for i in range(self.n):
            for j in range(self.n):
                if(not 0 in img[i][j] and not 255 in img[i][j]):
                    for channel in range(len(ret)):
                        ret[channel].append(img[i][j][channel])
        
        return ret
    
    @staticmethod
    def get_ga_list_spot(img: list) -> list:
        
        ret = [[0 for x in range(256)] for x in range(3)]

        for i in range(len(img)):
            for j in range(len(img[0])):
                if(not 0 in img[i][j] and not 255 in img[i][j]):
                    for channel in range(len(ret)):
                        ret[channel][img[i][j][channel]]+=1
        
        return ret
    
    def generate_img(self, apply_bias: bool):
        img = np.array([[self.skin_color for _ in range(self.n)] for _ in range(self.n)], dtype=np.uint8)
        cell_center = [self.n//2 + random.randint(-8, 8), self.n//2 + random.randint(-8, 8)]
        for i in range(self.n):
            for j in range(self.n):
                euclidean_distance = int(np.linalg.norm(cell_center-np.array([i, j])))
                if(euclidean_distance < self.spot_radius+1):
                    if(euclidean_distance > self.spot_radius-2):
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
        """Generate multiple images based on parameters. Images will be saved in
            ../img/folder_name as jsons, uncompressed."""
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


    @staticmethod
    def generate_dataset(zip_name : str) -> None:
        img_path = os.path.join('..', 'img')
        if(not os.path.exists(img_path)):
            return None
        dataset = []
        with zipfile.ZipFile(os.path.join(img_path, zip_name + '.zip'), 'r') as z:
            lst = z.namelist()
            for i in tqdm(range(len(lst))):
                filename = lst[i]
                if(filename[-1] != '/'):
                    with z.open(filename) as fl:
                        img = np.array(json.loads(fl.read().decode('utf-8')))
                        dataset.append([procImg.get_ga_list_spot(img), 'sick' in filename])
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
           