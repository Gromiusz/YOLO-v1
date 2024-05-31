import os
import shutil


def copy_lines_starting_with_small_numbers(labels_folder, class_number, class_counts=150):
    ''' Funkcja wyszukuje odpowiednie klasy a następnie tworzy nowe pliki labels zawierające tylko założoną ilość klas'''
    class_count = {i: 0 for i in range(class_number)} # słownik zliczeń klas
    for file_name in os.listdir(labels_folder):
        # if sum(class_count.values()) >= 250:
        #     break
        if all(count >= class_counts for count in class_count.values()):
            break 


        file_path = os.path.join(labels_folder, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".txt"):
            with open(file_path, "r") as file:
                lines = file.readlines()

            for line in lines:
                line_elements = line.strip().split()
                if len(line_elements) > 0 and line_elements[0].isdigit():
                    class_label = int(line_elements[0])
                    if class_label < class_number:
                        if class_count[class_label] < class_counts:
                            class_count[class_label] += 1
                            with open(os.path.join("labels", file_name), "a") as new_file: # Tworzy plik o tej samej nazwie w folderze labels i wkleja skopiowane linie
                                new_file.write(line)

            print(f"\rClass counts: {class_count}", end='', flush=True)  # Wyświetl aktualne zliczenia klas
    return class_count

def copy_images():
    ''' Kopiuje odpowiednie obrazy z oryginalnego folderu do roboczego '''
    copied_count = 0
    label_files_remaining = os.listdir("labels")
    for label_file in label_files_remaining:
        image_name = os.path.splitext(label_file)[0] + ".jpg"
        shutil.copy(f"images_ori/{image_name}", "images")
        copied_count += 1
        if copied_count % 10 == 0:
            print(f"\r{copied_count} images have been copied.", end='', flush=True)

def main():
    os.makedirs("images", exist_ok=True)
    os.makedirs("labels", exist_ok=True)

    print("Copiyng files with 10 first classes...")
    class_count = copy_lines_starting_with_small_numbers("labels_prep50", class_number=50)
    print("Processing completed")

    print("\nClass counts:", class_count)
    
    print("Copiyng images with same names as processed (.txt) files...")
    copy_images() # Kopiowanie tylko tych obrazów, których pliki txt są w folderze labels
    print("Copying completed")

    print("\nAll processes have been completed successfully.")

if __name__ == "__main__":
    main()



