import torchvision
import torch
import cv2 as cv
import numpy as np

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=2, pretrained_backbone=True)
model.load_state_dict(torch.load("./mask_rcnn_pedestrian_model.pt"))
model.eval()
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

# 使用GPU
train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    model.cuda()


def object_detection__demo():
    frame = cv.imread("D:/images/pedestrian_02.png")
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    print("frame", frame.shape)
    blob = transform(frame)
    c, h, w = blob.shape
    print(c, h, w)
    input_x = blob.view(1, c, h, w)
    output = model(input_x.cuda())[0]
    boxes = output['boxes'].cpu().detach().numpy()
    scores = output['scores'].cpu().detach().numpy()
    labels = output['labels'].cpu().detach().numpy()
    masks = output['masks'].cpu().detach().numpy()
    print("boxes", boxes.shape, "masks", masks.shape)
    index = 0
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    result_mask = np.zeros((h, w), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        if scores[index] > 0.9:
            print("score: ", scores[index])
            mask = np.reshape(masks[index], (514, 684))
            mask[mask >= 0.5] = 255
            mask[mask < 0.5] = 0
            result_mask = cv.add(result_mask, np.uint8(mask))
            cv.rectangle(frame, (np.int32(x1), np.int32(y1)), (np.int32(x2), np.int32(y2)), (0, 0, 255), 2, 8, 0)
            index += 1

    cv.imshow("Mask-RCNN Demo", frame)
    cv.imshow("mask", result_mask)
    result = cv.bitwise_and(frame, frame, mask=result_mask)
    cv.imshow("result", result)
    cv.bitwise_not(result_mask, result_mask)
    mm = cv.imread("D:/images/bg_01.jpg")
    m1 = cv.resize(mm, (w, h))
    cv.imshow("m1", m1)
    cv.waitKey(0)
    red_bg = cv.bitwise_and(m1, m1, mask=result_mask)
    cv.imshow("red_bg", red_bg)
    blend_red = cv.add(red_bg, result)
    cv.imshow("background red", blend_red)
    cv.imwrite("D:/pedestrian_02mask_rcnn.png", frame)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    object_detection__demo()