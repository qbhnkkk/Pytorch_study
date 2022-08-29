import cv2 as cv
import numpy as np
import pyttsx3
import os

emotion_labels = ["neutral", "anger", "disdain", "disgust", "fear", "happy", "sadness", "surprise"]
defect_labels = ["In", "Sc", "Cr", "PS", "RS", "Pa"]


def emotions_onnx_demo():
    emotion_net = cv.dnn.readNetFromONNX("face_emotions_model.onnx")
    image = cv.imread("D:/facedb/test/367.jpg")
    blob = cv.dnn.blobFromImage(image, 0.00392, (64, 64), (127, 127, 127), False) / 0.5
    emotion_net.setInput(blob)
    res = emotion_net.forward("output")
    idx = np.argmax(np.reshape(res, (8)))
    emotion_txt = emotion_labels[idx]
    cv.putText(image, emotion_txt, (10, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv.imshow("emotion detection", image)

    cv.waitKey(0)
    cv.destroyAllWindows()


def surface_defect_onnx_demo():
    surface_net = cv.dnn.readNetFromONNX("surface_defect_model_resnet18.onnx")
    root_dir = "D:/project/pytorch_stu/data/enu_surface_defect/test"
    fileNames = os.listdir(root_dir)
    for f in fileNames:
        image = cv.imread(os.path.join(root_dir, f))
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        # cv.dnn.blobFromImage(image , 1/255, resize大小, 255*mean) / std(均值)
        blob = cv.dnn.blobFromImage(image, 0.00392, (200, 200), (123, 116, 103), False) / 0.226
        surface_net.setInput(blob)
        res = surface_net.forward()
        idx = np.argmax(np.reshape(res, 6))
        # print(idx)
        surface_txt = defect_labels[idx]
        print(surface_txt, f)
        cv.putText(image, surface_txt, (10, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv.imshow("surface detection", image)
        cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    surface_defect_onnx_demo()
