import cv2
import numpy as np
import os

if __name__ == '__main__':
    originDataPath="Data/originData/"
    list=os.listdir(originDataPath)
    for child in list:
        s_list = os.listdir(originDataPath + child)
        for i in s_list:
            print(i)


