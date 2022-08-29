import cv2 as cv
import os
import numpy as np
from surface_defect_cnn import SurfaceDefectResNet
import torch
from torchvision import transforms

defect_labels = ["In", "Sc", "Cr", "PS", "RS", "Pa"]


def defect_demo():
    cnn_model = SurfaceDefectResNet()
    cnn_model.load_state_dict(torch.load("./surface_defect_model.pt"))
    cnn_model.eval()
    cnn_model.cuda()
    print(cnn_model)
    root_dir = "D:/project/pytorch_stu/data/enu_surface_defect/test"
    img_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225]),
                                        transforms.Resize((200, 200))
                                        ])
    fileNames = os.listdir(root_dir)
    for f in fileNames:
        image = cv.imread(os.path.join(root_dir, f))
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        x_input = img_transform(image).view(1, 3, 200, 200)
        # print(x_input.size())
        # img = cv.resize(image, (200, 200))
        # img = (np.float32(img) / 255.0 - 0.5) / 0.5
        # img = img.transpose((2, 0, 1))
        # x_input = torch.from_numpy(img).view( 1, 3, 200, 200)
        probs = cnn_model(x_input.cuda())
        predic_ = probs.view(6).cpu().detach().numpy()
        # print(predic_)
        idx = np.argmax(predic_)
        defect_txt = defect_labels[idx]
        print(defect_txt, f)
        cv.putText(image, defect_txt, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, 8)
        cv.imshow("defect_detection", image)
        cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    defect_demo()
