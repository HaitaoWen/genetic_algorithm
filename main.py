import numpy as np
import  xlrd as excel
from sklearn import svm
import matplotlib.pyplot as plt
import PCA


init_num = 4
#variation_rate = 0.01
iteration = 50
x_n = 11


def read_excel():#读表
    file = excel.open_workbook('/home/orion/Desktop/homework3/data/data.xls')
    table = file.sheets()[0]#获取表
    row_num = table.nrows - 1#求得人数
    data = np.empty((row_num, 7))#建立数据表
    data[:,0] = table.col_values(1)[1:]#类别
    data[:,1] = table.col_values(3)[1:]#身高
    data[:,2] = table.col_values(4)[1:]#体重
    data[:,3] = table.col_values(6)[1:]#数学
    data[:,4] = table.col_values(7)[1:]#文学
    data[:,5] = table.col_values(8)[1:]#运动
    data[:,6] = table.col_values(9)[1:]#模式识别
    return data

def normalize(data):#归一化处理
    h_min = data[:,1].min()
    h_max = data[:,1].max()
    w_min = data[:,2].min()
    w_max = data[:,2].max()
    data[:,1] = (data[:,1] - h_min)/(h_max - h_min)
    data[:,2] = (data[:, 2] - w_min)/(w_max - w_min)

def init_group():
    init_code = np.random.randint(1, 63, [1, init_num])
    code_bin = []
    for i in range(init_num):
        string = bin(init_code[0, i])[2:]
        N = 6 - len(string)
        zeros = '0' * N
        string = zeros + string
        code_bin.append(string)
    return code_bin

def seprate_data(code, data):
    code_bin = '1' + code
    cols = code_bin.count('1')
    son_data = np.empty((data.shape[0], cols))
    j = 0
    for i in range(data.shape[1]):
        if code_bin[i] == '1':
            son_data[:,j] = data[:,i]
            j += 1
    return son_data

def seprate_male_female(data):
    rows, cols = data.shape
    male_num = int(sum(data[:,0]))
    female_num = rows - male_num
    male_data = np.empty((male_num, cols))
    female_data = np.empty((female_num, cols))
    i ,j = 0, 0
    for k in range(rows):
        if data[k,0] == 1:
            male_data[i, :] = data[k, :]
            i += 1
        else:
            female_data[j, :] = data[k, :]
            j += 1
    return male_data, female_data

def fitness(male_data, female_data):
    male_num = male_data.shape[0]
    female_num = female_data.shape[0]
    p_m = male_num/(male_num + female_num)
    p_f = 1 - p_m
    m_var = male_data[:,1:].var(0)
    f_var = female_data[:,1:].var(0)
    m_var_sum = sum(m_var)
    f_var_sum = sum(f_var)
    Sw = p_m * m_var_sum + p_f * f_var_sum#类内距离

    m_mean = male_data[:,1:].mean(0)
    f_mean = female_data[:,1:].mean(0)
    mean = (m_mean * male_num + f_mean * female_num)/(male_num + female_num)
    Sb_m = sum((m_mean - mean) * (m_mean - mean))
    Sb_f = sum((f_mean - mean) * (f_mean - mean))
    Sb = Sb_m * p_m + Sb_f * p_f#类间距离

    J = Sb / Sw
    return J

def fitness_function(code, data):
    son_data = seprate_data(code, data)#分离数据
    male_data, female_data = seprate_male_female(son_data)#男女数据
    J = fitness(male_data, female_data)#计算适应度
    return J

def contrast(fitness):
    random_value = np.random.uniform(size=(1, init_num))
    index = np.empty((1,init_num))
    for i in range(init_num):
        num = random_value[0,i]
        for j in range(init_num):
            if num <= fitness[j]:
                index[0,i] = j
                break
    return index

def across(code):
    #i = np.random.randint(1,6)
    i = 3
    str_tem1 = code[0][:i]
    str_tem2 = code[0][i:]
    str_tem3 = code[2][:i]
    str_tem4 = code[2][i:]
    code[0] = str_tem1 + str_tem4
    code[2] = str_tem3 + str_tem2

    #i = np.random.randint(1, 6)
    i = 3
    str_tem1 = code[1][:i]
    str_tem2 = code[1][i:]
    str_tem3 = code[3][:i]
    str_tem4 = code[3][i:]
    code[1] = str_tem1 + str_tem4
    code[3] = str_tem3 + str_tem2
    return code

def max_index(fitness):
    m = 0
    index = 0
    for i in range(init_num):
        if m < fitness[0,i]:
            m = fitness[0,i]
            index = i
    return m, index

def variation(code):
    i = np.random.randint(0,4, dtype=int)
    j = np.random.randint(0,6, dtype=int)
    if code[i][j] == '1':
        n = 0
    else:
        n = 1
    code[i] = code[i][:j] + str(n) + code[i][j+1:]
    if code[i] == '000000':
        num = np.random.randint(1, 63, dtype=int)
        string = bin(num)[2:]
        zeros = '0'*(6 - len(string))
        code[i] = zeros + string

def svm_classifier(train_data, model):
    data = train_data[:,1:]
    label = train_data[:,0]
    model = model.fit(data, label)
    #model.score(data, label)
    return model

def evalution_performance(result, label):
    # 正样本 男生， 负样本 女生
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    rows = label.shape[0]
    for i in range(rows):
        if label[i] == 1:  # 正例
            if label[i] == result[i]:
                TP += 1
            else:
                FN += 1
        else:  # 反例
            if label[i] == result[i]:
                TN += 1
            else:
                FP += 1
    SE = round(float(TP)/float(TP + FN),4)
    TPR = SE
    FPR = round(float(FP)/float(TN + FP),4)
    SP = round(float(TN)/float(TN + FP),4)
    ACC = round(float(TP + TN)/float(FP + FP + TN + FN),4)
    return SE, SP, ACC, TPR, FPR

def array_rank(array):
    rows, cols = array.shape
    for i in range(cols):
        for j in range(i+1, cols):
            if array[0, i] > array[0, j]:
                t1 = array[0, i]
                t2 = array[1, i]
                array[0, i] = array[0, j]
                array[1, i] = array[1, j]
                array[0, j] = t1
                array[1, j] = t2

if __name__ == '__main__':
    data = read_excel()
    normalize(data)

#PCA降维
    sex = data[:,0]#性别列
    data = PCA.pca(data[:,1:])
    data = np.column_stack((sex, data))#添加性别列

#遗传算法选择特征
    # code = init_group()#初始化群体
    # fitness_max = 0
    # for k in range(iteration):
    #     code_fitness = np.empty((1, init_num))
    #     for i in range(init_num):
    #         code_fitness[0,i] = fitness_function(code[i], data)
    #     max_buffer, index_buffer = max_index(code_fitness)
    #     if fitness_max < max_buffer:
    #         fitness_max = max_buffer
    #         code_final = code[index_buffer]
    #     fitness_cumsum = (code_fitness/code_fitness.sum()).cumsum()
    #     index = contrast(fitness_cumsum)
    #     previous_code = code.copy()
    #     code = []
    #     for i in range(init_num):
    #          code.append(previous_code[int(index[0,i])])
    #     across(code)
    #     variation(code)
    # print(fitness_max, code_final)
    # data = seprate_data(code_final, data)

#SVM分类器分类
    # rows = data.shape[0]
    # results = np.empty((rows, 1))
    # svm_roc_list = np.empty((2, x_n))
    # SE_V = np.empty((1, x_n))
    # SP_V = np.empty((1, x_n))
    # ACC_V = np.empty((1, x_n))
    # th = np.linspace(0, 1, x_n)
    # model = svm.SVC(kernel='linear', C=1, gamma=1, probability=True)
    # print('BP\SVM\TREE 训练中.......')
    # for k in range(x_n):
    #     print('变阈值交叉验证中.....', 'threshold:', th[k])
    #     threshold = th[k]
    #     for i in range(rows):
    #         test_data = data[[i], :]
    #         train_data = np.delete(data, i, axis=0)
    #         model = svm_classifier(train_data, model)
    #         svm_predict = model.predict_proba(test_data[[0], 1:])[0][1]
    #         if svm_predict < threshold:
    #             svm_predict = 0
    #         else:
    #             svm_predict = 1
    #         results[i, 0] = svm_predict
    #     SE, SP, ACC, TPR, FPR = evalution_performance(results[:, 0], data[:, 0])
    #     if FPR == 0.0:
    #         TPR = 0
    #     print("SVM  ", 'SE:', SE, "SP:", SP, "ACC:", ACC,'TPR:',TPR,'FPR:',FPR)
    #     svm_roc_list[0,k] = FPR
    #     svm_roc_list[1,k] = TPR
    #     SE_V[0,k]=SE
    #     SP_V[0,k]=SP
    #     ACC_V[0,k]=ACC
    # array_rank(svm_roc_list)
    # plt.scatter(svm_roc_list[0, :], svm_roc_list[1, :], marker='x', c='#FFA500')
    # plt.xlabel('FPR')
    # plt.ylabel('TPR')
    # plt.legend(['ROC'], loc='lower right')
    # plt.plot(svm_roc_list[0, :], svm_roc_list[1, :], '#FFA500')
    # plt.show()
    # svm_auc = np.trapz(svm_roc_list[1, :], svm_roc_list[0, :])
    # print('SVM_AUC:',round(svm_auc,4))
    #
    # plt.scatter(th, SE_V[0, :], marker='*', c='b')
    # plt.xlabel('threshold')
    # plt.ylabel('SE')
    # plt.legend(['SE'], loc='upper right')
    # plt.plot(th, SE_V[0, :], 'b')
    # plt.show()
    #
    # plt.scatter(th, SP_V[0, :], marker='*', c='g')
    # plt.xlabel('threshold')
    # plt.ylabel('SP')
    # plt.legend(['SP'], loc='lower right')
    # plt.plot(th, SP_V[0, :], 'g')
    # plt.show()
    #
    # plt.scatter(th, ACC_V[0, :], marker='*', c='r')
    # plt.xlabel('threshold')
    # plt.ylabel('ACC')
    # plt.legend(['ACC'], loc='upper right')
    # plt.plot(th, ACC_V[0, :], 'r')
    # plt.show()
    #
    # print('整体测试中........')
    # SVM_error_num = 0
    # threshold = 0.4
    # for i in range(rows):
    #     svm_predict = model.predict_proba(data[[i], 1:])[0][1]
    #     if svm_predict < threshold:
    #         svm_predict = 0
    #     else:
    #         svm_predict = 1
    #     if svm_predict != data[i, 0]:
    #         SVM_error_num += 1
    # print('total_num:', rows, 'SVM_true_rate:', round(1.0 - float(SVM_error_num) / float(rows), 3) * 100.0, '%')
    # print('ROC', svm_roc_list)
    # print('SE', SE_V)
    # print('SP', SP_V)
    # print('ACC', ACC_V)