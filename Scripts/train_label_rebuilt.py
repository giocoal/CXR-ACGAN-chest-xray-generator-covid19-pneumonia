import os

def create_image_label_file(img_folder_path, label_file_path):

    with open(label_file_path, 'w') as f:
        for folder in os.listdir(img_folder_path):
            folder_path = os.path.join(img_folder_path, folder)
            if os.path.isdir(folder_path):
                for image in os.listdir(folder_path):
                    image_path = os.path.join(folder_path, image)
                    if os.path.isfile(image_path):
                        f.write(f"{image} {folder}\n")

if __name__ == '__main__':
    img_folder_path = 'C:/Users/marco/Desktop/Local_Documents/data/COVIDx-splitted-resized-112_augm/train'
    label_file_path = 'C:/Users/marco/Desktop/Local_Documents/data/COVIDx-splitted-resized-112_augm/train_COVIDx9A_rebuild.txt'
    create_image_label_file(img_folder_path, label_file_path)