import argparse
from typing import List, Tuple, Any

import cv2
import numpy as np
import skvideo.io
import torch
from PIL import Image, ImageEnhance
from google.colab.patches import cv2_imshow

from constants import NMS_TH, PRED_TH, DEVICE
from fish_dataset import FishDataLoaders
from fish_net import FishNet


def nms(
        bboxes: List[Tuple[Any, Any, Any]], data_loaders: FishDataLoaders, iou_threshold: float = NMS_TH,
        pred_threshold: float = PRED_TH):
    bboxes = [box for box in bboxes if box[1] > pred_threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []
    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [box for box in bboxes if FishNet.calc_iou(
            chosen_box[2].clone().detach(),
            box[2].clone().detach()
        ) < iou_threshold
                  ]

        bboxes_after_nms.append(chosen_box)

    bboxes_after_nms = [box for box in bboxes_after_nms if box[0] != data_loaders.datasets.samples['background']]

    return bboxes_after_nms


def test_model(model_path: str, video_paths: List[str], frames_dir: str):
    model = torch.load(model_path, map_location=DEVICE)

    dataloaders = FishDataLoaders(frames_dir=frames_dir)
    colors = np.random.uniform(0, 255, size=(dataloaders.num_classes, 3))

    for path in video_paths:
        vidcap = skvideo.io.vreader(path)
        count = 0
        model.eval()
        for frame in vidcap:
            if not count % 500:
                print(f'Frame number: {count}')
                sharpner = ImageEnhance.Sharpness(Image.fromarray(frame))
                frame = sharpner.enhance(8)
                img = torch.FloatTensor(np.array(frame)).to(DEVICE) / 255
                img = img.unsqueeze(0).permute((0, 3, 1, 2))
                detections = model(img)[0]
                threshhold = detections['scores'][:8][-1] if len(detections['scores']) > 0 else 0
                detections = list(zip(detections['labels'], detections['scores'], detections['boxes']))
                detections = nms(detections, dataloaders, pred_threshold=threshhold, iou_threshold=0.3)
                if detections:
                    print(f'Printing frame number: {count}')
                    print_image(np.array(frame), detections, dataloaders, colors)
            count += 1


def print_image(image, detections: List[Tuple[Any, Any, Any]], dataloaders: FishDataLoaders, colors: np.array):
    orig = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)

    # loop over the detections
    for label, confidence, boxes in detections:
        if confidence >= 0.1:
            idx = int(label)
            box = boxes.detach().cpu().numpy()
            species = dataloaders.idx_to_species[idx]
            print(box, species, confidence)
            (startX, startY, endX, endY) = box.astype('int')

            if species == 'background':
                continue
            cv2.rectangle(orig, (startX, startY), (endX, endY),
                          colors[idx], 1)
            y = startY if startY > 0 else endY
            x = startX if startX > 0 else endX
            # b = box.astype(int)

            cv2.putText(orig, species, (x, y), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255), 1)
    cv2_imshow(orig)

    cv2.waitKey(0)
