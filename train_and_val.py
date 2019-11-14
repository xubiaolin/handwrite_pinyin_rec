# encoding=utf-8
import cv2
import numpy as np
import os
import random
import shutil
import json


def knn(k, testdata, traindata, labels):
    '''定义算法'''
    traindatasize = traindata.shape[0]
    dif = np.tile(testdata, (traindatasize, 1)) - traindata
    sqrdif = dif ** 2
    sumsqrdif = sqrdif.sum(axis=1)
    distance = sumsqrdif ** 0.5
    sorted_distance = distance.argsort()
    count = {}
    for i in range(0, k):
        vote = labels[sorted_distance[i]]
        count[vote] = count.get(vote, 0) + 1
    sorted_count = sorted(count.items(), key=lambda x: x[1], reverse=True)
    return sorted_count[0][0]


def img2Model(originDataPath, modelpath):
    list = os.listdir(originDataPath)
    for child in list:
        s_list = os.listdir(originDataPath + child)
        for i in s_list:

            filepath=originDataPath + child + '/' + i
            print(filepath)
            img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8),0)
            img = cv2.resize(img, (32, 32))
            img = np.asarray(img)
            img[img > 127] = 255
            img[img <= 127] = 1
            img[img == 255] = 0
            dstFileName = modelPath + i.split('.')[0] + '.txt'
            np.savetxt(dstFileName, img, fmt='%d', delimiter=' ')


# 加载数据
def load_data(dataFilePath):
    arr = np.loadtxt(dataFilePath, dtype=np.int)
    arr = arr.flatten()
    return arr


# 建立训练数据集
def makeTrainData(trainpath):
    labels = []
    trainfile = os.listdir(trainpath)

    trainarr = np.zeros((len(trainfile), 1024))
    for i in range(0, len(trainfile)):
        # print(trainfile[i])
        thislabel = trainfile[i].split(".")[0].split("_")[0]

        if len(thislabel) != 0:
            labels.append(int(thislabel))
        trainarr[i, :] = load_data(trainpath + trainfile[i])
    return trainarr, labels


# 随机分拣出测试集，其他文件为训练集
def shutildata(modelpath, trainpath, testpath):
    txtlist = os.listdir(modelpath)
    index = [random.randint(0, len(txtlist)) for i in range(10)]
    # print(index)
    arr = [txtlist[i].split('.')[0].split("_")[1] for i in index]
    for i in txtlist:
        try:
            if i.split(".")[0].split("_")[1] in arr:
                shutil.copy(modelpath + "/" + i, testpath)
            else:
                shutil.copy(modelpath + "/" + i, trainpath)
        except:
            pass


# 验证
def validate(testpath, trainpath, k):
    trainarr, labels = makeTrainData(trainpath)
    testfiles = os.listdir(testpath)
    count = 0

    # 读取字典表
    with open('num_char.json', 'r') as f:
        dict = json.loads(f.read())
        # print(dict)

    for i in range(0, len(testfiles)):
        testpicname = testfiles[i].split("_")[0]
        testarr = load_data(testpath + testfiles[i])
        result = knn(k, testarr, trainarr, labels)

        testpicname = dict[str(testpicname)]
        result = dict[str(result)]

        print("真正字母:"+testfiles[i] +"  " + testpicname + "  " + "测试结果为:{}".format(result))
        if str(testpicname) == str(result):
            count += 1
    print("-----------------------------")
    print("测试集为:{}个,其中正确了{}个".format(len(testfiles),count))
    print("正确率为{}".format(count / len(testfiles)))
    print()


trainPath = 'Data/train/'
testPath = 'Data/test/'
modelPath = 'Data/model/'
originalData = 'Data/originData/'


def main(method):
    if method == "train":
        shutil.rmtree(trainPath)
        shutil.rmtree(testPath)
        shutil.rmtree(modelPath)

        os.mkdir(testPath)
        os.mkdir(trainPath)
        os.mkdir(modelPath)

        img2Model(originalData, modelPath)
        shutildata(modelPath, trainPath, testPath)
        validate(testPath, trainPath, 4)

    if method == 'pridict':
        validate(testPath, trainPath, 5)


if __name__ == '__main__':
    main('train')
