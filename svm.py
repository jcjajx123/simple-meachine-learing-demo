import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
'''
预处理fisheriris数据集
根据demo只保留前100行SepalLength，SepalWith的数据
将'setosa'定为正类，将'versicolor'定为负类,qu
将SepalLength，SepalWith赋值给训练数据data；+，-赋值给标签

'''
def loadfile():
    df = pd.read_csv('fisheriris.csv')
    df.drop(['PetalLength'], axis=1, inplace=True)
    df.drop(['PetalWidth'], axis=1, inplace=True)
    df.drop(df.index[100:150], inplace=True)
    df.replace('setosa', 1, inplace=True)
    df.replace('versicolor', -1, inplace=True)
    # print(df)
    dataset = []
    for i in range(100):
        dataset.append([df.iat[i, 0], df.iat[i, 1],df.iat[i, 2]])
    dataset = np.array(dataset)
    positive = np.array([[0, 0, 0]])  # +1的样本集
    negative = np.array([[0, 0, 0]])  # -1的样本集
    for i in range(dataset.shape[0]):
        if (dataset[i][2] == 1):
            positive = np.row_stack((positive, np.array([dataset[i]])))
        else:
            negative = np.row_stack((negative, np.array([dataset[i]])))
    return positive[1:, :], negative[1:, :], dataset

#核函数
def kernel(xi,xj):
    #sigma=10
    #if ke=='linear':
    return xi.dot(xj.T)
    #elif ke=='rbf':
    '''
    M = xi.shape[0]
    K = np.zeros((M, 1))
    for l in range(M):
        A = np.array([xi[l]]) - xj
        K[l] = [np.exp(-0.5 * float(A.dot(A.T)) / (sigma ** 2))]
    return K
    '''
#非边界点的集合
def findnonbound(alpha,C):
    nonbound=[]
    for i in range(len(alpha)):
        if(0<alpha[i] and alpha[i]<C):
            nonbound.append(i)
    return nonbound

def selectJrand(i,N):#随机选择j
    j=i
    while(j==i):
        j=int(np.random.uniform(0,N))
    return j

class SVM():
    def __init__(self,data,label,C,eps):
        self.X=data
        self.Y=label
        self.C=C
        self.eps=eps
        self.m=self.X.shape[0]
        self.b=0
        self.alpha=np.zeros((self.m,1)) #拉格朗日乘子
        self.E = np.zeros((self.m, 2))  # 误差缓存表m*2，第一列为更新状态（0-未更新，1-已更新），第二列为缓存值

    #计算Ei
    def getE(self,i):
        xi = np.array([self.X[i]])
        y = np.array([self.Y]).T
        gx = float(self.alpha.T.dot(y * kernel(self.X, xi))) + self.b
        Ei = gx - self.Y[i]
        return Ei

    # 更新缓存项Ei包括计算Ei和设置对应的更新状态为1
    def updateEi(self,i):
        Ei=self.getE(i)
        self.E[i]=[1,Ei]

    def selectaJ(self, i, Ei):  # 内循环，根据i选择j
        self.E[i] = [1, Ei]  # 更新Ei
        validE = np.nonzero(self.E[:, 0])[0]  # validE保存更新状态为1的缓存项的行指标
        if (len(validE) > 1):
            j = 0
            maxDelta = 0
            Ej = 0
            for k in validE:  # 寻找最大的|Ei-Ej|
                if (k == i):
                    continue
                Ek = self.getE(k)
                if (abs(Ei - Ek) > maxDelta):
                    j = k
                    maxDelta = abs(Ei - Ek)
                    Ej = Ek
        else:  # 随机选择
            j = selectJrand(i, self.m)
            Ej = self.getE(j)
        return j, Ej
    #循环 先选择第一个变量alphai 再通过selectaJ选取第二个变量alphaj
    def inner(self, i):
        Ei = self.getE(i)
        if ((self.Y[i] * Ei > self.eps and float(self.alpha[i]) > 0) or
                (self.Y[i] * Ei < -self.eps and float(self.alpha[i]) < self.C)):  # alpha[i]违反了KKT条件
            # Ei=gx-self.Y[i] 故Ei*self.Y[i]=yigx-1  KKT条件为 alpha>0时 yigx-1<=0 alpha
            j, Ej = self.selectaJ(i,Ei)  # 选择对应的alpha[j]
            alphaiold = float(self.alpha[i])
            alphajold = float(self.alpha[j])
            if (self.Y[i] != self.Y[j]):
                L = max(0, alphajold - alphaiold)
                H = min(self.C, self.C + alphajold - alphaiold)
            else:
                L = max(0, alphajold + alphaiold - self.C)
                H = min(self.C, alphajold + alphaiold)
            if (L == H): return 0
            xi = np.array([self.X[i]])
            xj = np.array([self.X[j]])
            eta = float(kernel(xi, xi) + kernel(xj, xj) - 2 * kernel(xi, xj))
            if (eta <= 0): return 0
            alphajnewunc = alphajold + self.Y[j] * (Ei - Ej) / eta  # 未剪辑的alphajnew
            # 更新alphaj
            if (alphajnewunc > H):
                self.alpha[j] = [H]
            elif (alphajnewunc < L):
                self.alpha[j] = [L]
            else:
                self.alpha[j] = [alphajnewunc]
            # 更新Ej
            self.updateEi(j)
            if (abs(float(self.alpha[j]) - alphajold) < 0.00001): return 0
            # 更新alphai
            self.alpha[i] = [alphaiold + self.Y[i] * self.Y[j] * (alphajold - float(self.alpha[j]))]
            # 更新b
            bi = -Ei - self.Y[i] * float(kernel(xi, xi)) * (float(self.alpha[i]) - alphaiold) - \
                 self.Y[j] * float(kernel(xj, xi)) * (float(self.alpha[j]) - alphajold) + self.b
            bj = -Ej - self.Y[i] * float(kernel(xi, xj)) * (float(self.alpha[i]) - alphaiold) - \
                 self.Y[j] * float(kernel(xj, xj)) * (float(self.alpha[j]) - alphajold) + self.b

            if (0 < float(self.alpha[i]) and float(self.alpha[i]) < self.C):
                self.b = bi
            elif (0 < float(self.alpha[j]) and float(self.alpha[j]) < self.C):
                self.b = bj
            else:
                self.b = 0.5 * (bi + bj)
            # 更新Ei  Ei的更新要求放在b的更新之后
            self.updateEi(i)
            return 1
        else:
            return 0

    def visualize(self, positive, negative):
        plt.xlabel('X1')  # 横坐标
        plt.ylabel('X2')  # 纵坐标
        plt.scatter(positive[:, 0], positive[:, 1], c='r', marker='o')  # +1样本红色标出
        plt.scatter(negative[:, 0], negative[:, 1], c='g', marker='o')  # -1样本绿色标出
        nonZeroAlpha = self.alpha[np.nonzero(self.alpha)]  # 非零的alpha对应的点即是支持向量 非支持向量在模型中不起作用 对应的alpha必为0
        supportVector = self.X[np.nonzero(self.alpha)[0]]  # 支持向量
        y = np.array([self.Y]).T[np.nonzero(self.alpha)]  # 支持向量对应的标签
        plt.scatter(supportVector[:, 0], supportVector[:, 1], s=100, c='b', alpha=0.5, marker='o')  # 标出支持向量
        print("支持向量个数:", len(nonZeroAlpha))
        X1 = np.arange(0, 8, 0.1)
        X2 = np.arange(0, 8, 0.1)
        x1, x2 = np.meshgrid(X1, X2)
        g = self.b
        for i in range(len(nonZeroAlpha)):
            g+=nonZeroAlpha[i]*y[i]*(x1*supportVector[i][0]+x2*supportVector[i][1])
        plt.contour(x1, x2, g, 0, colors='y')  # 画出超平面
        plt.show()


def SMO(X, Y, C, eps, maxIters):  # SMO的主程序
    SVMClassifier = SVM(X, Y, C, eps)
    iters = 0
    iterEntire = True  # 由于alpha被初始化为零向量，所以先遍历整个样本集
    while (iters < maxIters):  # 循环在整个样本集与非边界点集上切换，达到最大循环次数时退出
        iters += 1
        if (iterEntire):  # 循环遍历整个样本集
            alphaPairChanges = 0
            for i in range(SVMClassifier.m):  # 外层循环
                alphaPairChanges += SVMClassifier.inner(i)
            if (alphaPairChanges == 0):
                break  # 整个样本集上无alpha对变化时退出循环
            else:
                iterEntire = False  # 有alpha对变化时遍历非边界点集
        else:  # 循环遍历非边界点集
            alphaPairChanges = 0
            nonbound = findnonbound(SVMClassifier.alpha, SVMClassifier.C)  # 非边界点集
            for i in nonbound:  # 外层循环
                alphaPairChanges += SVMClassifier.inner(i)
            if (alphaPairChanges == 0):
                iterEntire = True  # 非边界点全满足KKT条件，则循环遍历整个样本集
    return SVMClassifier


if __name__ == "__main__":
    positive, negative, dataset = loadfile()  # 返回+1与-1的样本集，总训练集
    X = dataset[:, 0:2]  # X1,X2
    Y = dataset[:, 2]  # Y
    SVMClassifier = SMO(X, Y, 1, 0.1, 100)
    SVMClassifier.visualize(positive, negative)




















































