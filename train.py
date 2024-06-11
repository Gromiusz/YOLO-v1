"""
Main file for training Yolo model on Pascal VOC dataset

"""

import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import TrafficSignsDataset
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)
from loss import YoloLoss

seed = 123
torch.manual_seed(seed)

LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8  # 16 # 64
WEIGHT_DECAY = 0
EPOCHS = 10
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
TEST_MODEL = True
LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = "data/images"
LABEL_DIR = "data/labels"
TEST_IMG_DIR = "test_data/images_test"
TEST_LABEL_DIR = "test_data/labels_test"


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(), ])


def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss) / len(mean_loss)}")

def precision(tp, fp):
    if tp + fp == 0:
        return 0
    return tp / (tp + fp)

def recall(tp, fn):
    if tp + fn == 0:
        return 0
    return tp / (tp + fn)

def f1_score(prec, rec):
    if prec + rec == 0:
        return 0
    return 2 * (prec * rec) / (prec + rec)

def calculate_statistics(predicted_boxes, true_boxes, iou_threshold):
    true_positives = 0
    detected_boxes = []
    false_positives = len(predicted_boxes)

    for true_box in true_boxes:
        for pred_box in predicted_boxes:
            iou = intersection_over_union(torch.tensor(pred_box[2:]), torch.tensor(true_box[1:]))
            if iou > iou_threshold and pred_box[0] == true_box[0] and pred_box not in detected_boxes:
                true_positives += 1
                detected_boxes.append(pred_box)
                break

    false_positives -= true_positives
    false_negatives = len(true_boxes) - true_positives

    return true_positives, false_positives, false_negatives

def test_fn(test_loader, model):
    with torch.no_grad():
        model.eval()
        loop = tqdm(test_loader, leave=True)
        total_true_positives, total_false_positives, total_false_negatives = 0, 0, 0

        for batch_idx, (x, y) in enumerate(loop):
            x = x.to(DEVICE)
            predictions = model(x)
            for idx in range(x.size(0)):
                predicted_boxes = cellboxes_to_boxes(predictions[idx].unsqueeze(0))
                predicted_boxes = non_max_suppression(predicted_boxes[0], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
                true_boxes = cellboxes_to_boxes(y[idx].unsqueeze(0))
                true_boxes = [[box[0], box[1], box[2], box[3], box[4], box[5]] for box in true_boxes[0]]

                tp, fp, fn = calculate_statistics(predicted_boxes, true_boxes, iou_threshold=0.5)
                total_true_positives += tp
                total_false_positives += fp
                total_false_negatives += fn

        prec = precision(total_true_positives, total_false_positives)
        rec = recall(total_true_positives, total_false_negatives)
        f1 = f1_score(prec, rec)

        print(f"Precyzja: {prec:.4f}, Czułość: {rec:.4f}, F1-Score: {f1:.4f}")
def main():

    print("Creating model ...\n")
    model = Yolov1(split_size=7, num_boxes=2, num_classes=50).to(DEVICE)  # Zmienione na 50 klas
    optimizer = optim.Adam(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    print("Initialization loss function...\n")
    loss_fn = YoloLoss()

    if LOAD_MODEL or TEST_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE, map_location=torch.device('cpu')), model, optimizer)

    if not TEST_MODEL:
        train_dataset = TrafficSignsDataset(
            "data/data.csv",
            transform=transform,
            img_dir=IMG_DIR,
            label_dir=LABEL_DIR,
        )
        print("Loading training dataset...\n")
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            shuffle=True,
            drop_last=True,
        )
    else:
        test_dataset = TrafficSignsDataset(
            "test_data/data_test.csv", transform=transform, img_dir=TEST_IMG_DIR, label_dir=TEST_LABEL_DIR,
        )
        print("Loading testing dataset...\n")
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            shuffle=True,
            drop_last=True,
        )
        print("Loading testing dataset...")
        # test_fn(test_loader, model)

    print("Starting first epoch...")
    for epoch in range(EPOCHS):
        print("Epoch " + str(epoch) + "\n")
        if TEST_MODEL:
            print("No training, loading model ...")
            for x, y in test_loader:
                x = x.to(DEVICE)
                for idx in range(BATCH_SIZE):  # Assuming the batch size can vary
                    bboxes = cellboxes_to_boxes(model(x[idx].unsqueeze(0)))  # Process one image at a time
                    bboxes = non_max_suppression(bboxes[0], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
                    # Convert tensor outputs to list and adjust dimensions for plotting
                    bboxes = [[bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]] for bbox in bboxes]
                    plot_image(x[idx].permute(1, 2, 0).to("cpu"), bboxes)

            import sys
            sys.exit()

        pred_boxes, target_boxes = get_bboxes(train_loader, model, iou_threshold=0.5, threshold=0.4)
        mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint",
                                               num_classes=50)

        print(f"Train mAP: {mean_avg_prec}")

        if mean_avg_prec > 0.01:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
            import time
            time.sleep(10)

        train_fn(train_loader, model, optimizer, loss_fn)


if __name__ == "__main__":
    main()
