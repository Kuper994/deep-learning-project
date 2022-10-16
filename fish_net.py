import argparse
import copy
import math
import os
from collections import Counter
from typing import List, Tuple, Any

import numpy as np
import torch
import subprocess

import torchvision
from torch import optim
from torchvision.models import detection
from tqdm import tqdm

from data_preparation import prepare_data
from fish_dataset import FishDataLoaders
from constants import DEVICE

try:
    import imgaug as ia
    import imgaug.augmenters as iaa
    from imgaug import parameters as iap
    from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
except ImportError:
    subprocess.check_call(['pip', 'install', 'imgaug'])
    import imgaug as ia
    import imgaug.augmenters as iaa
    from imgaug import parameters as iap
    from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


class FishNet:
    def __init__(self, dataloaders: FishDataLoaders, num_classes: int, n_trainable: int = 2):
        self.model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
            weights=detection.RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT,
            trainable_backbone_layers=1).to(DEVICE)

        # Freeze layers:
        for p in list(self.model.head.classification_head.parameters())[: -n_trainable]:
            p.requires_grad = False

        for p in list(self.model.head.regression_head.parameters())[: -n_trainable]:
            p.requires_grad = False

        # replace classification layer
        in_features = self.model.head.classification_head.cls_logits.in_channels
        num_anchors = self.model.head.classification_head.num_anchors
        self.model.head.classification_head.num_classes = num_classes

        cls_logits = torch.nn.Conv2d(in_features, num_anchors * num_classes, kernel_size=3, stride=1, padding=1).to(
            DEVICE)
        torch.nn.init.normal_(cls_logits.weight, std=0.01)
        torch.nn.init.constant_(cls_logits.bias, -math.log((1 - 0.01) / 0.01))
        torch.nn.init.normal_(self.model.head.regression_head.bbox_reg.weight, std=0.01)
        torch.nn.init.zeros_(self.model.head.regression_head.bbox_reg.bias)

        # assign cls head to model
        self.model.head.classification_head.cls_logits = cls_logits

        # Decay LR by a factor of 0.1 every 7 epochs
        # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)

        self.dataloaders = dataloaders

    @staticmethod
    def calc_iou(true_boxes: torch.Tensor, pred_boxes: torch.Tensor, eps: float = 1e-6):
        box1_x1 = true_boxes[..., 0]
        box1_y1 = true_boxes[..., 1]
        box1_x2 = true_boxes[..., 2]
        box1_y2 = true_boxes[..., 3]
        box2_x1 = pred_boxes[..., 0]
        box2_y1 = pred_boxes[..., 1]
        box2_x2 = pred_boxes[..., 2]
        box2_y2 = pred_boxes[..., 3]

        x1 = torch.max(box1_x1, box2_x1)
        y1 = torch.max(box1_y1, box2_y1)
        x2 = torch.min(box1_x2, box2_x2)
        y2 = torch.min(box1_y2, box2_y2)
        intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
        box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
        box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

        return intersection / (box1_area + box2_area - intersection + eps)

    @staticmethod
    def get_avg_precisions(
            all_ground_truths: List[Tuple[int, Any, Any]], all_detections: List[Tuple[int, Any, Any, Any]],
            iou_threshold: float, num_classes: int, eps: float):
        best_gt_idx = 0
        average_precisions = []
        for c in range(num_classes):
            detections = []
            ground_truths = []

            for detection in all_detections:
                if detection[1] == c:
                    detections.append(detection)

            for true_box in all_ground_truths:
                if true_box[1] == c:
                    ground_truths.append(true_box)

            amount_bboxes_ = Counter([gt[0] for gt in ground_truths])
            amount_bboxes = dict()

            for key, val in amount_bboxes_.items():
                amount_bboxes[key] = torch.zeros(val)

            TP = torch.zeros((len(detections)))
            FP = torch.zeros((len(detections)))
            total_true_bboxes = len(ground_truths)

            if total_true_bboxes == 0:
                continue

            for detection_idx, detection in enumerate(detections):
                # Only take out the ground_truths that have the same
                # training idx as detection
                ground_truth_img = [
                    bbox for bbox in ground_truths if bbox[0] == detection[0]
                ]

                best_iou = 0

                for idx, gt in enumerate(ground_truth_img):
                    iou = FishNet.calc_iou(torch.tensor(detection[3]), torch.tensor(gt[2]))

                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx
                if best_iou > iou_threshold:
                    # only detect ground truth detection once
                    if amount_bboxes[detection[0]][best_gt_idx] == 0:
                        # true positive and add this bounding box to seen
                        TP[detection_idx] = 1
                        amount_bboxes[detection[0]][best_gt_idx] = 1
                    else:
                        FP[detection_idx] = 1

                # if IOU is lower then the detection is a false positive
                else:
                    FP[detection_idx] = 1

            tp_cumsum = torch.cumsum(TP, dim=0)
            fp_cumsum = torch.cumsum(FP, dim=0)
            recalls = tp_cumsum / (total_true_bboxes + eps)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + eps)
            precisions = torch.cat((torch.tensor([1]), precisions))
            recalls = torch.cat((torch.tensor([0]), recalls))
            # torch.trapz for numerical integration
            average_precisions.append(torch.trapz(precisions, recalls))
        return average_precisions

    @staticmethod
    def calc_map(
            all_ground_truths: List[Tuple[int, Any, Any]], all_detections: List[Tuple[int, Any, Any, Any]],
            num_classes: int, eps=1e-6):
        # Ground truths - the boxes tagged by the lab, list of tuples of the sort: (img_id, sample_label, box)
        # detections - all the detections outputted by the model on the tested set,
        # list of: (img_id, sample_label, score, box)

        # The code for the mAP score was taken from:
        # https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/metrics/mean_avg_precision.py

        map_per_th = {}
        for iou_threshold in [0.1, 0.3, 0.5, 0.8]:
            print('threshold: ', iou_threshold)
            average_precisions = FishNet.get_avg_precisions(all_ground_truths, all_detections, iou_threshold,
                                                            num_classes, eps)
            map_per_th[iou_threshold] = sum(average_precisions) / (len(average_precisions) + eps)

        return map_per_th

    def train(self):
        self.model.train()
        self.dataloaders.mode = 'train'

    def eval(self, mode: str = 'val'):
        self.model.eval()
        self.dataloaders.mode = mode

    def eval_model(self, mode: str = 'val'):
        with torch.set_grad_enabled(False):
            self.eval(mode)
            pred_imgs = []
            true_imgs = []
            true_boxes = []
            true_labels = []
            all_detections = []
            for i, inputs in enumerate(self.dataloaders.dataloader):
                images = inputs['img'].to(DEVICE)
                annots = inputs['annot'].squeeze(0).to(DEVICE)
                labels = []
                for label in inputs['label']:
                    label = label.to(DEVICE)
                    labels.extend(label.unbind())
                true_boxes.extend(annots.unbind())
                true_labels.extend(labels)
                true_imgs.extend([i] * len(labels))
                outputs = self.model(images)[0]  # batch size is 1
                pred_imgs.extend([i] * len(outputs['boxes']))
                detections = list(
                    sorted(zip(pred_imgs, outputs['labels'], outputs['scores'], outputs['boxes']), key=lambda x: x[1],
                           reverse=True))

                all_detections.extend(detections)
            ground_truth = list(zip(true_imgs, true_labels, true_boxes))
            map_per_th = self.calc_map(ground_truth, all_detections, num_classes=self.dataloaders.num_classes)
        self.train()
        return map_per_th

    def train_model(self, num_epochs: int = 25, learning_rate: float = 1e-3):
        print("device is:", DEVICE)

        optimizer = \
            optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=learning_rate, amsgrad=True)
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_loss = -np.inf
        self.train()
        train_losses = []
        val_maps = []
        test_maps = []

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            running_loss = 0.0
            for inputs in tqdm(self.dataloaders.dataloader):
                images = inputs['img'].to(DEVICE)
                annots = inputs['annot'].to(DEVICE)
                labels = [l.to(DEVICE) for l in inputs['label']]

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                targets = []
                with torch.set_grad_enabled(True):
                    for b, l in zip(annots, labels):
                        b = torch.stack(b.unbind()[:l.shape[0]])
                        targets.append({'boxes': b.unsqueeze(0) if len(b.shape) == 1 else b, 'labels': l})

                    loss = self.model(images, targets)

                    # backward + optimize only if in training phase
                    loss_sum = 0.1 * loss['bbox_regression'] + loss['classification']
                    loss_sum.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                    optimizer.step()

                    # statistics
                    running_loss += loss_sum.item() * len(inputs)
            # scheduler.step()

            epoch_loss = running_loss / self.dataloaders.dataset_sizes['train']

            val_map_dict = self.eval_model('val')
            test_map_dict = self.eval_model('test')

            train_losses.append(epoch_loss)
            val_maps.append(val_map_dict[0.3].item())
            test_maps.append(test_map_dict[0.3].item())

            print(f'Train Loss: {epoch_loss:.4f}')
            print(f'Validation mAP Scores: {val_map_dict}')
            print(f'Test mAP Scores: {test_map_dict}')

            if val_map_dict[0.1] >= best_loss:
                best_loss = val_map_dict[0.1]
                best_model_wts = copy.deepcopy(self.model.state_dict())

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        return self.model, train_losses, val_maps, test_maps


def train_fishnet(data_filename: str, data_dir: str = 'raw_data', frames_dir: str = 'frames',
                  is_converted: bool = False, data_types: int = 0, use_augs: bool = False, num_augs: int = 10,
                  num_epochs: int = 25, output_file: str = '', learning_rate: float = 1e-3, trainable_layers: int = 2):
    if output_file and os.path.dirname(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_path = os.path.join(frames_dir, data_filename)
    if not os.path.exists(df_path):
        prepare_data(data_dir=data_dir, frames_dir=frames_dir, is_converted=is_converted,
                     data_types=data_types, create_augs=use_augs, num_augs=num_augs,
                     to_create_frames=True, to_create_bg_frames=True)
    train_path = 'augs_data.csv' if use_augs else 'train.csv'
    data_loaders = FishDataLoaders(frames_dir=frames_dir, train_path=train_path)
    fishnet = FishNet(dataloaders=data_loaders, num_classes=data_loaders.num_classes, n_trainable=trainable_layers)
    fishnet.train_model(num_epochs=num_epochs, learning_rate=learning_rate)
    if output_file:
        torch.save(fishnet.model, output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data arguments:
    parser.add_argument('-f', '--frames-dir', help='Frames Directory Name', default='frames')
    parser.add_argument('-d', '--data-dir', help='Raw Data Directory Name', default='raw_data')
    parser.add_argument('--use-augs', help='Use augmentations when training', action='store_true')
    parser.add_argument('--is-converted', help='Using the converted videos', action='store_true')
    parser.add_argument('--data-types', help='Choose between using only the samples (2), only the background (1) '
                                             'or using both (0), default is 0.', default=0, type=int, choices=[0, 1, 2])
    parser.add_argument('--num-augs', help='Number of augmentations for each relevant frame', default=10, type=int)

    # model arguments
    parser.add_argument('--trainable-layers', help='Number of trainable layers, for all layers - enter 0.',
                        default=2, type=int)
    parser.add_argument('--num-epochs', help='Number of epochs to train the model.', default=25, type=int)
    parser.add_argument('--lr', help='Learning rate.', default=1e-3, type=float)
    parser.add_argument('--output-file', help='Output path to save the trained model in.', default='')

    # Read arguments from command line
    args = parser.parse_args()

    filename = 'data.csv' if args.data_types == 1 else 'background.csv' \
        if args.data_types == 2 else 'data_and_background.csv'

    train_fishnet(data_filename=filename, data_dir=args.data_dir, frames_dir=args.frames_dir,
                  is_converted=args.is_converted, data_types=args.data_types, use_augs=args.use_augs,
                  num_augs=args.num_augs, num_epochs=args.num_epochs, output_file=args.output_file,
                  learning_rate=args.lr, trainable_layers=args.trainable_layers)
