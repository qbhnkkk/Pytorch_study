import cv2 as cv
import numpy as np

image = cv.imread("lena.jpg")
cv.imshow("input", image)
print(image.shape)
h, w, c = image.shape  # 原本顺序h w c
print(h, w, c)

# 维度转换为c h w
blob = np.transpose(image, (2, 0, 1))
print(blob.shape)

# 转化为浮点数
fi = np.float32(image) / 255.0  # 转为0~1之间的浮点数
cv.imshow('fi', fi)

# 转为灰度图像
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow('gray', gray)

# # 显示第0个通道数据
# cv.imshow('image[:,:,0]', image[:, :, 0])
# # 显示第1个通道数据
# cv.imshow('image[:,:,1]', image[:, :, 1])
# # 显示第2个通道数据
# cv.imshow('image[:,:,1]', image[:, :, 2])
# # 显示彩色图像
# cv.imshow('image1', image[:, :, :])

# 改变图像尺寸
dst = cv.resize(image, (256, 256))
# 缩小
# cv.imshow('zoom out', dst)
# 放大
dst2 = cv.resize(image, (1024, 1024))
# cv.imshow('zoom in', dst2)

# 视频
# cap = cv.VideoCapture(0)  # 0表示调用电脑摄像头 文件则输入路径
# while True:
#     ret, frame = cap.read()
#     # 不需要某个返回则
#     # _, frame = cap.read()
#     if ret is not True:
#         break
#     cv.imshow('frame', frame)
#     c = cv.waitKey(1)  # 1表示每一帧之间停留1ms 如果要放慢调大值
#     if c == 27:  # Esc
#         break

# 截取roi
box = [200, 200, 200, 200]  # x,y,w,h
roi = image[200:400, 200:400, :]  # 高度方向 ，宽度方向 ，通道数
cv.imshow('roi', roi)

# 创建空白图像
m1 = np.zeros((512, 512), dtype=np.uint8)  # 初始化矩阵
cv.imshow("m1", m1)
# 彩色空白图像
m2 = np.zeros((512, 512, 3), dtype=np.uint8)  # 初始化矩阵
m2[:, :] = [0, 255, 0]  # BGR
cv.imshow("m2", m2)

# 改变通道顺序 bgr -> rgb
rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
cv.imshow('rgb', rgb)

# 画矩形框
cv.rectangle(image, (200, 200), (400, 400), (0, 0, 255), 2, 8)
cv.imshow('input', image)

cv.waitKey(0)
cv.destroyAllWindows()
