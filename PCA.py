import numpy as np

dimension = 5

def pca(data):
    data1 = data.copy()
    mean = np.mean(data1, axis=0)#求每一列的均值
    data1 = data1 - mean#去均值化
    cov = np.cov(data1, rowvar= 0)#将每一列看作一个变量，求各列之间的协方差
    print(cov)
    n, x = np.linalg.eig(cov)
    print(n)
    index = np.argsort(n)
    index = index[:-(dimension+1):-1]
    w = x[:,index]
    truncated = np.dot(data, w)
    return  truncated

