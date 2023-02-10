from keras.preprocessing.image import ImageDataGenerator
import cv2
import os
import numpy as np

def generate_images(directory, class_folder, save_directory, total_images, batch_size):
    
    # Data augmentation function
    datagen = ImageDataGenerator(
                                rotation_range=10,
                                width_shift_range=0.1,
                                height_shift_range=0.1,
                                horizontal_flip=True,
                                brightness_range=(0.9, 1.1),
                                zoom_range=(0.85, 1.15),
                                fill_mode='constant',
                                cval=0.)
    
    # Load the images for each class
    generator = datagen.flow_from_directory(
                                            directory,
                                            target_size=(112, 112),
                                            class_mode='binary',
                                            classes=[class_folder],
                                            batch_size= batch_size,
                                            shuffle=False)
    
    # Loop for generate images starting from 'generator' and using 'datagen'
    i = 0
    for batch in generator:
        if i + batch_size > total_images:
            break
        i += 1
        x = batch[0]
        file_path = generator.filenames[generator.batch_index - 1]
        file_name = os.path.basename(file_path)
        save_path = f"{save_directory}/{os.path.splitext(file_name)[0]}_aug_{i}.png"
        cv2.imwrite(save_path, x[0])
        print(i, save_path)


if __name__ == '__main__':

    '''
    For the input directory, enter the path main: dataset/train/

    For the output directory, enter the folder-specific path: dataset_aug/train/classes/
    '''

    #Normal
    dir_input_normal = 'C:/Users/marco/Desktop/Local_Documents/data/COVIDx-splitted-resized-112-process-augm/train/'
    dir_output_normal = 'C:/Users/marco/Desktop/Local_Documents/data/COVIDx-splitted-resized-112-process-augm/train/normal'
    generate_images(dir_input_normal, "normal", dir_output_normal, 7689, 1)

    # #Pneuomia
    # dir_input_pneumonia = 'C:/Users/marco/Desktop/Local_Documents/data/COVIDx-splitted-resized-112-process/train/'
    # dir_output_pneuomina = 'C:/Users/marco/Desktop/Local_Documents/data/COVIDx-splitted-resized-112-process-augm/train/pneumonia'
    # generate_images(dir_input_pneumonia, "pneumonia", dir_output_pneuomina, 3, 1)