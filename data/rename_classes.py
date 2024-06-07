import os

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def read_file(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
    return lines

def write_file(filepath, lines):
    with open(filepath, 'w') as file:
        file.writelines(lines)

def save_label_map(label_map, filepath):
    with open(filepath, 'w') as file:
        for original_label, new_label in label_map.items():
            file.write(f"{original_label} {new_label}\n")

def process_file(filepath, label_map, next_label):
    lines = read_file(filepath)
    new_lines = []
    for line in lines:
        parts = line.split()
        original_label = int(parts[0])
        if original_label not in label_map:
            label_map[original_label] = next_label
            next_label += 1
        new_label = label_map[original_label]
        new_line = f"{new_label} " + " ".join(parts[1:]) + "\n"
        new_lines.append(new_line)
    return new_lines, next_label

def main():
    ensure_directory('labels_prep50')

    label_map = {}
    next_label = 0

    files = sorted([f for f in os.listdir('labels_ori') if f.endswith('.txt')])
    total_files = len(files)

    for index, filename in enumerate(files):
        input_filepath = os.path.join('labels_ori', filename)
        output_filepath = os.path.join('labels_prep50', filename)

        new_lines, next_label = process_file(input_filepath, label_map, next_label)
        write_file(output_filepath, new_lines)

        remaining_files = total_files - (index + 1)
        print(f"Processed '{filename}'. {remaining_files} files remaining.")

    highest_new_number = next_label - 1
    print(f"The highest new number is {highest_new_number}.")

    save_label_map(label_map, 'label_map.txt')

if __name__ == "__main__":
    main()
