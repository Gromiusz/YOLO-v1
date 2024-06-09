import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from model import Yolov1
from dataset import TrafficSignsDataset
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    load_checkpoint
)
from loss import YoloLoss

# Settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_CLASSES = 50  # Assuming 50 classes based on your dataset definition
CONF_THRESHOLD = 0.4
NMS_IOU_THRESHOLD = 0.5
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0
NUM_WORKERS = 2
PIN_MEMORY = True

# Transform for the dataset
transform = transforms.Compose([
    transforms.Resize((448, 448)),  # Adjust if your model expects a different size
    transforms.ToTensor(),
])

# Load the test dataset
test_dataset = TrafficSignsDataset(
    csv_file="test_data/test_data.csv", 
    img_dir="test_data/images_test", 
    label_dir="test_data/labels_test", 
    transform=transform
)
test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            shuffle=True,
            drop_last=True,
        )

# Define your YOLO model class (Assuming YOLOv1 is defined elsewhere)
model = Yolov1(split_size=7, num_boxes=2, num_classes=NUM_CLASSES).to(DEVICE)
optimizer = optim.Adam(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
load_checkpoint(torch.load("overfit.pth.tar"), model, optimizer)

# Function to test the model
def test_model(loader, model, iou_threshold, threshold, device):
    model.eval()
    pred_boxes, target_boxes = get_bboxes(test_loader, model, iou_threshold=0.5, threshold=0.4)

    mean_avg_prec = mean_average_precision(
        pred_boxes, target_boxes, iou_threshold=iou_threshold, box_format="midpoint"
    )

    print(f"mAP: {mean_avg_prec}")
    model.train()

    return mean_avg_prec

# Main script execution
if __name__ == "__main__":
    mean_avg_prec = test_model(test_loader, model, iou_threshold=NMS_IOU_THRESHOLD, threshold=CONF_THRESHOLD, device=DEVICE)
    print(f"Mean Average Precision: {mean_avg_prec}")
