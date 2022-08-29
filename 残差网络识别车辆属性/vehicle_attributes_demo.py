from __future__ import print_function
import cv2 as cv
import numpy as np
import torch
import logging as log
from vehicle_attributes_cnn import ResidualBlock, VehicleAttributesResNet
from openvino.inference_engine import IENetwork, IECore

color_labels = ["white", "gray", "yellow", "red", "green", "blue", "black"]
type_labels = ["car", "bus", "truck", "van"]

model_dir = "D:/project/pytorch_stu/model/intel/vehicle-detection-adas-0002/FP32"
model_xml = model_dir + "vehicle-detection-adas-0002.xml"
model_bin = model_dir + "vehicle-detection-adas-0002.bin"

net = IENetwork(model=model_xml, weights=model_bin)

ie = IECore()
log.info("Device info:")
versions = ie.get_versions("CPU")

cnn_model = torch.load("./vehicle_attributes_model.pt")
print(cnn_model)

input_blob = next(iter(net.inputs))
n, c, h, w = net.inputs[input_blob].shape

capture = cv.VideoCapture("D:/images/video/cars-1900.mp4")
ih = capture.get(cv.CAP_PROP_FRAME_HEIGHT)
iw = capture.get(cv.CAP_PROP_FRAME_WIDTH)

input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))

log.info("Loading model to the device")
exec_net = ie.load_network(network=net, device_name="CPU")
log.info("Creating infer request and starting inference")

while True:
    ret, src = capture.read()
    if ret is not True:
        break
    images = np.ndarray(shape=(n, c, h, w))
    images_hw = []
    ih, iw = src.shape[:-1]
    images_hw.append((ih, iw))
    if (ih, iw) != (h, w):
        image = cv.resize(src, (w, h))
    image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    images[0] = image
    res = exec_net.infer(inputs={input_blob: images})

    # 解析SSD输出内容
    res = res[out_blob]
    license_score = []
    license_boxes = []
    data = res[0][0]
    index = 0
    for number, proposal in enumerate(data):
        if proposal[2] > 0.75:
            ih, iw = images_hw[0]
            label = np.int(proposal[1])
            confidence = proposal[2]
            xmin = np.int(iw * proposal[3])
            ymin = np.int(ih * proposal[4])
            xmax = np.int(iw * proposal[5])
            ymax = np.int(ih * proposal[6])

            if xmin < 0:
                xmin = 0
            if ymin < 0:
                ymin = 0
            if xmax >= iw:
                xmax = iw - 1
            if ymax >= ih:
                ymax = ih - 1
            vehicle_roi = src[ymin:ymax, xmin:xmax,:]
            img = cv.resize(vehicle_roi, (72, 72))
            img = (np.float32(img) / 255.0 - 0.5) / 0.5
            img = img.transpose((2, 0, 1))
            x_input = torch.from_numpy(img).view(1, 3, 72, 72)
            color_, type_ = cnn_model(x_input.cuda())
            predict_color = torch.max(color_, 1)[1].cpu().detach().numpy()[0]
            predict_type = torch.max(type_, 1)[1].cpu().detach().numpy()[0]
            attrs_txt = "color:%s, type:%s"%(color_labels[predict_color], type_labels[predict_type])
            cv.rectangle(src, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            cv.putText(src, attrs_txt, (xmin, ymin), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv.imshow("Vehicle Attributes Recognition Demo", src)
    res_key = cv.waitKey(1)
    if res_key == 27:
        break