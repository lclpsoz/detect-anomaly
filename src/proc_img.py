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
    
    def imread(self):
    
        return None
    
    def gaussian_otsu_threshold_conv(self, img):
        
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        blur = cv.GaussianBlur(gray, (5, 5), 0)
        ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        return cv.bitwise_not(thresh)
    
    def img_dilation(self, thresh):
        
        kernel = np.ones((5, 5, np.uint8))
        
        return cv.dilate(thresh,kernel, iterations=1)
    
    def find_countours_img(self, dilation):

        contours, hierarchy = cv.findContours(dilation, cv.RETR_TREE,cv.CHAIN_APPROX_TC89_KCOS)

        return contours
    
    def get_biggest_lesion(self, contours):
        
        cnt = []
        contour_area = 0
        
        for x in range(len(contours)):
            
            if cv.contourArea(contours[x]) > contour_area:
                contour_area = cv.contourArea(contours[x])
                cnt = contours[x]
        if len(cnt.shape) == 3:
            cnt = [cnt]
        
        return cnt
    
    def rect_img(self, cnt, img):
        
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
    
    def draw_countors_img(self, img, cnt, rgb: tuple, thicc: int):
        
        cv.drawContours(img, cnt, -1, rgb, thicc)

        return
    
    def grabcut_img(self, img, rect):
        
        mask = np.zeros(img.shape[:2], np.uint8)

        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        
        cv.grabCut(img,mask,rect,bgdModel,fgdModel,5, cv.GC_INIT_WITH_RECT)
        
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        
        return img*mask2[:,:,np.newaxis]

           