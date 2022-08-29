import torch
import cv2 as cv
import numpy as np
from landmark_cnn import Net, ChannelPool

model_bin = "D:\project\pytorch_stu\model\opencv_face_detector_uint8.pb";
config_text = "D:\project\pytorch_stu\model\opencv_face_detector.pbtxt";


def image_landmark_demo():
    cnn_model = torch.load("./model_landmarks.pt")
    image = cv.imread("1164.jpg")
    cv.imshow("input", image)
    h, w, c = image.shape
    img = cv.resize(image, (64, 64))
    img = (np.float32(img) / 255.0 - 0.5) / 0.5
    img = img.transpose((2, 0, 1))
    x_input = torch.from_numpy(img).view(1, 3, 64, 64)
    probs = cnn_model(x_input.cuda())
    lm_pts = probs.view(5, 2).cpu().detach().numpy()
    print(lm_pts)
    for x, y in lm_pts:
        print(x, y)
        x1 = x * w
        y1 = y * h
        cv.circle(image, (np.int32(x1), np.int32(y1)), 2, (0, 0, 255), 2, 8, 0)
    cv.imshow("face_land_mark_demo", image)
    cv.imwrite("D:/landmark_det_result.png", image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def video_landmark_demo():
    cnn_model = torch.load("./model_landmarks.pt")
    capture = cv.VideoCapture(0, cv.CAP_DSHOW)
    # capture = cv.VideoCapture("D:/project/data/video/example_dsh.mp4")

    # load tensorflow model
    net = cv.dnn.readNetFromTensorflow(model_bin, config=config_text)
    while True:
        ret, frame = capture.read()
        if ret is not True:
            break
        frame = cv.flip(frame, 1)
        h, w, c = frame.shape
        blobImage = cv.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False);
        net.setInput(blobImage)
        cvOut = net.forward()
        # 绘制检测矩形
        for detection in cvOut[0, 0, :, :]:
            score = float(detection[2])
            if score > 0.5:
                left = detection[3] * w
                top = detection[4] * h
                right = detection[5] * w
                bottom = detection[6] * h

                # roi and detect landmark
                roi = frame[np.int32(top):np.int32(bottom), np.int32(left):np.int32(right), :]
                rw = right - left
                rh = bottom - top
                img = cv.resize(roi, (64, 64))
                img = (np.float32(img) / 255.0 - 0.5) / 0.5
                img = img.transpose((2, 0, 1))
                x_input = torch.from_numpy(img).view(1, 3, 64, 64)
                probs = cnn_model(x_input.cuda())
                lm_pts = probs.view(5, 2).cpu().detach().numpy()
                for x, y in lm_pts:
                    x1 = x * rw
                    y1 = y * rh
                    cv.circle(roi, (np.int32(x1), np.int32(y1)), 2, (0, 0, 255), 2, 8, 0)

                # 绘制
                cv.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), thickness=2)
                cv.putText(frame, "score:%.2f" % score, (int(left), int(top)), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                           (0, 0, 255), 1)
                c = cv.waitKey(1)
                if c == 27:
                    break
                cv.imshow("face detection + landmark", frame)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    video_landmark_demo()
    # image_landmark_demo()
