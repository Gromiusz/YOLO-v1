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
EPOCHS = 1000
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
TEST_MODEL = False
LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = "data/images"
LABEL_DIR = "data/labels"


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


def main():
    # if not LOAD_MODEL and not TEST_MODEL:
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
            "data/data_test.csv", transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR,
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

    print("Starting first epoch...")
    for epoch in range(EPOCHS):
        if TEST_MODEL:
            print("No training, loading model ...")
            for x, y in test_loader:
                x = x.to(DEVICE)
                # for idx in range(8):
                #     bboxes = cellboxes_to_boxes(model(x))
                #     bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
                #     plot_image(x[idx].permute(1, 2, 0).to("cpu"), bboxes)
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
