import os
import csv

'''
Program służy do generowania pliku csv, w którym pliki z danymi (lables) przypisywane są do odpowiednich obrazów
'''
images_folder = "images/"
labels_folder = "labels/"

image_files = os.listdir(images_folder)

# Pary - nazwa obrazu i pliku tekstowego
data = [(image_file, image_file.split('.')[0] + '.txt') for image_file in image_files]

csv_filename = "data.csv"

with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['image', 'text'])
    for image, text in data:
        writer.writerow([image, text])

print(f"CSV file has been generated: {csv_filename}")

for image, text in data:
    image_path = os.path.join(images_folder, image)
    text_path = os.path.join(labels_folder, text)
    print(f"{image_path} -> {text_path}")
