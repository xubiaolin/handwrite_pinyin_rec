import os
import cv2
import numpy as np
path_a = 'C:\\Users\\MarkXu\\Desktop//i/'


if __name__ == '__main__':
    count = 0
    path = path_a

    list = os.listdir(path)
    for i in list:
        print(path+i)
        os.rename(path+i, path+"5_"+str(count)+".jpg")

        # img = cv2.imread(path + i,0)
        # img=np.asarray(img)
        # img[img>127]=255
        # img[img<=127]=0
        # img=cv2.resize(img,(32,32))
        # os.remove(path + i)
        # cv2.imwrite(path+"5_"+str(count)+".jpg", img)

        count += 1

