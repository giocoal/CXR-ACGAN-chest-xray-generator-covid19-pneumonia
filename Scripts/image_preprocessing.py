import cv2
import numpy as np
import os
import glob2 as glob

'''

Data pre-processing details:

The COVIDx dataset was used to train all tested deep neural network architectures.
As a pre-processing step, the chest CXR images were cropped (top 8% of the image) prior to training in order to mitigate commonly-found embedded 
textual information in the CXR images.

'''

def crop_top(img, percent=0.15):
    offset = int(img.shape[0] * percent)
    return img[offset:]

def central_crop(img):
    size = min(img.shape[0], img.shape[1])
    offset_h = int((img.shape[0] - size) / 2)
    offset_w = int((img.shape[1] - size) / 2)
    return img[offset_h:offset_h + size, offset_w:offset_w + size]


'''
This function applies the Pre-processing functions over the whole train dataset
'''

def process_and_save_images(dataset_dir, output_dir, size, top_percent=0.08, crop=True):
    if not os.path.exists(dataset_dir):
        print(f"Nessuna directory trovata: {dataset_dir}")
        return
    classes = os.listdir(dataset_dir)
    saves = 0
    fail = 0 
    # Cicla nelle sottocartelle 
    for class_name in classes:
        # print(class_name)
        class_dir = os.path.join(dataset_dir, class_name)
        # print(class_dir)
        if not os.path.isdir(class_dir): # controlla se le directory sono valide
            continue
        output_class_dir = os.path.join(output_dir, class_name) # Crea la directory di salvataggio
        # print(output_class_dir)
        image_files = os.listdir(class_dir) # Salva tutte le directory complete con le immagini
        # print(image_files)
        for image_file in image_files: # itera per ogni immagine e va applicare le funzioni
            image_path = os.path.join(class_dir, image_file)
            img = cv2.imread(image_path)
            # print(img.shape)
            img = crop_top(img, percent=top_percent)
            if crop:
                img = central_crop(img)
            img = cv2.resize(img, (size, size))
            output_path = os.path.join(output_class_dir, image_file)
            print(output_path)
            save = cv2.imwrite(output_path, img)
            if save:
                # print(f"Immagine salvata: {output_path}")
                saves +=1
            else:
                # (f"Immagine non saltata: {output_path}")
                fail +=1
    print('Numero totali di immagini procesate e salvate',  saves)
    print('Numero di immagini non salvate', fail)


if __name__ == '__main__':
    dataset_dir = 'C:\\Users\\marco\\Desktop\\Local_Documents\\data\\COVIDx-splitted-resized-112\\train' # Directory MAIN output
    output_dir = 'C:\\Users\\marco\\Desktop\\Local_Documents\\data\\COVIDx-splitted-resized-112_process\\train' # Directory MAIN output
    process_and_save_images(dataset_dir=dataset_dir, output_dir=output_dir, size=1024, top_percent=0.08)