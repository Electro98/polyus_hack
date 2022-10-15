from typing import Callable

import albumentations
import albumentations.pytorch
import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from config import *


def get_model_instance_segmentation(num_classes):
    """Созданий модели."""

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def create_model(model_path: str = MODEL_PATH, treshold: float = THRESHOLD) -> Callable[[np.ndarray], np.ndarray]:
    """Загружает модель для детектирования негабарита."""

    if not torch.cuda.is_available():
        raise RuntimeError("cuda is not available now")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 2
    model = get_model_instance_segmentation(num_classes)

    try:
        model.load_state_dict(torch.load(model_path))
        print("model loaded successfully")
    except RuntimeError:
        print("model not loaded, sad")

    model.to(device)

    model.eval()

    get_transform = albumentations.Compose([
        albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        albumentations.pytorch.transforms.ToTensorV2(),
    ])

    cpu_device = torch.device("cpu")

    def predictor(frame: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            image = get_transform(image=frame)["image"].to(device)

            outputs = model([image])

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

            if len(outputs[0]['boxes']) != 0:
                boxes = outputs[0]['boxes'].data.numpy()
                scores = outputs[0]['scores'].data.numpy()
                boxes = boxes[scores >= treshold].astype(np.int32)
                draw_boxes = boxes.copy()

                for box in draw_boxes:
                    cv2.rectangle(frame,
                                (int(box[0]), int(box[1])),
                                (int(box[2]), int(box[3])),
                                (0, 0, 255), 3)
                    cv2.putText(frame, 'stone',
                                (int(box[0]), int(box[1]-5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),
                                2, lineType=cv2.LINE_AA)
                return frame
            return frame
    return predictor
