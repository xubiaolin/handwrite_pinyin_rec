import cv2
import numpy as np

if __name__ == '__main__':
    img=cv2.imread("1.jpg",0)
    img=np.asarray(img)
    cols=np.sum(img,0)
    rows=np.sum(img,1)
    print(cols)