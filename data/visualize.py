import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def visualize_bounding_boxes(image_path, label_path, class_number, displayed_classes):
    image = Image.open(image_path)
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    classes_on_image = set()

    with open(label_path, "r") as label_file:
        lines = label_file.readlines()
        for line in lines:
            line_elements = line.strip().split()
            if len(line_elements) >= 5:
                class_label = int(line_elements[0])
                if class_label < class_number:
                    x, y, width, height = map(float, line_elements[1:5])

                    x *= image.width
                    y *= image.height
                    width *= image.width
                    height *= image.height

                    rect = patches.Rectangle((x - width / 2, y - height / 2), width, height, linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                    ax.text(x - width / 2, y - height / 2, str(class_label), fontsize=8, color='r')
                    classes_on_image.add(class_label)

    if classes_on_image.issubset(displayed_classes):
        plt.close(fig)
        return

    displayed_classes.update(classes_on_image)

    plt.show()

def main():
    images_folder = "images_ori"
    labels_folder = "labels"
    label_files = os.listdir(labels_folder)

    class_number = 20  
    displayed_classes = set()

    for label_file in label_files:
        image_name = os.path.splitext(label_file)[0] + ".jpg"
        image_path = os.path.join(images_folder, image_name)

        if os.path.isfile(image_path):
            label_path = os.path.join(labels_folder, label_file)
        
            visualize_bounding_boxes(image_path, label_path, class_number, displayed_classes)

if __name__ == "__main__":
    main()
