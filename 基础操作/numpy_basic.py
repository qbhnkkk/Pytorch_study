import numpy as np

a = np.array([1, 2, 3, 4, 5, 6])
b = np.array([8, 7, 6, 5, 4, 3])
print("输出a,b矩阵的维度: a.shape= \n", a.shape, "b.shape=", b.shape)  # 维度
print(a)
aa = np.reshape(a, (3, 2))  # 三行两列
bb = np.reshape(b, (1, 1, 1, 6))
print("a矩阵转化为2*3的矩阵aa：np.reshape(a, (3, 2))=\n", aa.shape)
print("b矩阵转化为1*1*1*6的矩阵bb：np.reshape(b, (1, 1, 1, 6))=\n", bb.shape)

b1 = np.squeeze(bb)  # 矩阵的降维
print("bb矩阵的降维：np.squeeze(bb)=\n", b1.shape)

index = np.argmax(b1)  # 寻找最大值 返回最大值的index
print("find b1 max:np.argmax(b1)=\n", b1, index, b1[index])

index2 = np.argmax(bb)  # 寻找最大值 返回最大值的index
print("find bb max (false): bb[index2]=\n ", bb, index2, bb[index2])  # 错误
print("find bb max (true): bb[0][0][0][index2]=\n", bb, index2, bb[0][0][0][index2])  # 正确

aaa = aa.transpose((1, 0))  # 矩阵的转置 第一个维度和第二个维度的交换 从3*2 ——> 2*3
a2 = np.reshape(aa, -1)  # -1表示变为一维
print("矩阵的转置 第一个维度和第二个维度的交换 从3*2 ——> 2*3: aa.transpose((1, 0))=", aaa, aaa.shape)
print("-1表示变为一维:np.reshape(aa, -1)=\n", a2, a2.shape)

m1 = np.zeros((6, 6), dtype=np.uint8)  # 初始化矩阵
print("初始化矩阵:np.zeros((6, 6), dtype=np.uint8)=\n", m1)

m2 = np.linspace(6, 10, 100)  # 从低值6开始均值到10 分成100个均值
print("从低值6开始均值到10 分成100个均值:np.linspace(6, 10, 100)=\n", m2)
