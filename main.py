from config import *

import cv2
import os
from tqdm import tqdm
import numpy as np
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import warnings
warnings.filterwarnings("ignore")

import torch
import torchvision

import albumentations
import albumentations.pytorch


def get_model_instance_segmentation(num_classes):
    """ Созданий модели. """

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def main(dir_images: str):
    """ dir_images: директория с фотографиями. """

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 2
    model = get_model_instance_segmentation(num_classes)

    try:
        model.load_state_dict(torch.load(MODEL_PATH))
        print("Модель успешно загружена")
    except:
        print("Модель не загрузилась :(")

    model.to(device)

    get_transform = albumentations.Compose([
        albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        albumentations.pytorch.transforms.ToTensorV2()
    ])

    test_images = os.listdir(dir_images)

    cpu_device = torch.device("cpu")

    model.eval()
    with torch.no_grad():
        for i, image in tqdm(enumerate(test_images), total=len(test_images)):

            orig_image = cv2.imread(f"{dir_images}/{test_images[i]}", cv2.IMREAD_COLOR)
            image = get_transform(image=orig_image)["image"].to(device)
            # image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
            # # make the pixel range between 0 and 1
            # image /= 255.0
            # image = np.transpose(image, (2, 0, 1)).astype(np.float)
            # image = torch.tensor(image, dtype=torch.float)
            # image = torch.unsqueeze(image, 0)
            
            outputs = model([image])

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            if len(outputs[0]['boxes']) != 0:
                for counter in range(len(outputs[0]['boxes'])):
                    boxes = outputs[0]['boxes'].data.numpy()
                    scores = outputs[0]['scores'].data.numpy()
                    boxes = boxes[scores >= THRESHOLD].astype(np.int32)
                    draw_boxes = boxes.copy()

                for box in draw_boxes:
                    cv2.rectangle(orig_image,
                                (int(box[0]), int(box[1])),
                                (int(box[2]), int(box[3])),
                                (0, 0, 255), 3)
                    cv2.putText(orig_image, 'light',
                                (int(box[0]), int(box[1]-5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),
                                2, lineType=cv2.LINE_AA)
                cv2.imwrite(f"{SAVE_DIR_IMG}/{test_images[i]}", orig_image)
    print('TEST PREDICTIONS COMPLETE')


def create_model():
    """ dir_images: директория с фотографиями. """
    if torch.cuda.is_available():
        print("CUDA IS ALIVE")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 2
    model = get_model_instance_segmentation(num_classes)

    # try:
    model.load_state_dict(torch.load(MODEL_PATH))
    print("Модель успешно загружена")
    # except e:
        # print("Модель не загрузилась :(")

    model.to(device)

    model.eval()

    get_transform = albumentations.Compose([
        albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        albumentations.pytorch.transforms.ToTensorV2()
    ])

    cpu_device = torch.device("cpu")

    def predictor(frame: np.ndarray) -> np.ndarray:
        with torch.no_grad():
        # for i, image in tqdm(enumerate(test_images), total=len(test_images)):

            # orig_image = cv2.imread(f"{dir_images}/{test_images[i]}", cv2.IMREAD_COLOR)
            # image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
            image = get_transform(image=frame)["image"].to(device)

            outputs = model([image])

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

            if len(outputs[0]['boxes']) != 0:
                for counter in range(len(outputs[0]['boxes'])):
                    boxes = outputs[0]['boxes'].data.numpy()
                    scores = outputs[0]['scores'].data.numpy()
                    boxes = boxes[scores >= THRESHOLD].astype(np.int32)
                    draw_boxes = boxes.copy()

                for box in draw_boxes:
                    cv2.rectangle(frame,
                                (int(box[0]), int(box[1])),
                                (int(box[2]), int(box[3])),
                                (0, 0, 255), 3)
                    cv2.putText(frame, 'light',
                                (int(box[0]), int(box[1]-5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),
                                2, lineType=cv2.LINE_AA)
                # cv2.imwrite(f"{SAVE_DIR_IMG}/{test_images[i]}", orig_image)
                return frame
            return frame
    return predictor


if __name__ == "__main__":
    main("images/")
