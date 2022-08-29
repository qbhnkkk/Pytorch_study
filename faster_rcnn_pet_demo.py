import torchvision
import torch
import cv2 as cv
import numpy as np
num_classes = 3

coco_names = {'0': 'unknown', '1': 'dog', '2': 'cat'}
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=num_classes, pretrained_backbone=True)
model.load_state_dict(torch.load("./faster_rcnn_pet_model.pt"))
model.eval()
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

# 使用GPU
train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    model.cuda()


def pet_image_detection(): 
    image = cv.imread("D:/images/test.jpg")
    blob = transform(image)
    c, h, w = blob.shape
    input_x = blob.view(1, c, h, w)
    output = model(input_x.cuda())[0]
    boxes = output['boxes'].cpu().detach().numpy()
    scores = output['scores'].cpu().detach().numpy()
    labels = output['labels'].cpu().detach().numpy()
    print(boxes.shape, scores.shape, labels.shape)
    index = 0
    for x1, y1, x2, y2 in boxes:
        if scores[index] > 0.5:
            cv.rectangle(image, (np.int32(x1), np.int32(y1)),
                         (np.int32(x2), np.int32(y2)), (140, 199, 0), 4, 8, 0)
            label_id = labels[index]
            label_txt = coco_names[str(label_id)]
            cv.putText(image, label_txt, (np.int32(x1), np.int32(y1)), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1)
        index += 1
    cv.imshow("Faster-RCNN Pet Detection", image)
    cv.imwrite("D:/pet2.png", image)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    pet_image_detection()
