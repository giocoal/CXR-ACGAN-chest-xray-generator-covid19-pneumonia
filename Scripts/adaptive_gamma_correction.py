import cv2
import numpy as np
import os
import glob2 as glob
import pandas as pd

'''
The following functions implement the Adaptive Gamma Correction halo:

The image_agcwd function is the heart of Adaptive Gamma Correction. The function receives an image and the parameters a and truncated_cdf, 
which determine the shape of the correction. The function calculates the histogram of the image, normalises the cumulative distribution 
and the probability distribution. It then performs a transformation on the probability distribution, calculates the inverse cumulative distribution 
and uses it to correct the brightness of the image.

Finally, the process_bright and process_dimmed functions invoke the image_agcwd function with the appropriate parameters to process 
the bright and dark images, respectively

'''

def image_agcwd(img, a=0.25, truncated_cdf=False):
    h,w = img.shape[:2]
    hist,bins = np.histogram(img.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf.max()
    prob_normalized = hist / hist.sum()

    unique_intensity = np.unique(img)
    intensity_max = unique_intensity.max()
    intensity_min = unique_intensity.min()
    prob_min = prob_normalized.min()
    prob_max = prob_normalized.max()
    
    pn_temp = (prob_normalized - prob_min) / (prob_max - prob_min)
    pn_temp[pn_temp>0] = prob_max * (pn_temp[pn_temp>0]**a)
    pn_temp[pn_temp<0] = prob_max * (-((-pn_temp[pn_temp<0])**a))
    prob_normalized_wd = pn_temp / pn_temp.sum() # normalize to [0,1]
    cdf_prob_normalized_wd = prob_normalized_wd.cumsum()
    
    if truncated_cdf: 
        inverse_cdf = np.maximum(0.5,1 - cdf_prob_normalized_wd)
    else:
        inverse_cdf = 1 - cdf_prob_normalized_wd
    
    img_new = img.copy()
    for i in unique_intensity:
        img_new[img==i] = np.round(255 * (i / 255)**inverse_cdf[i])
   
    return img_new


def process_bright(img):
    img_negative = 255 - img
    agcwd = image_agcwd(img_negative, a=0.25, truncated_cdf=False)
    reversed = 255 - agcwd
    return reversed

def process_dimmed(img):
    agcwd = image_agcwd(img, a=0.75, truncated_cdf=True)
    return agcwd



'''
This function applies the Adaptive Gamma Correction over the whole test dataset
'''

def apply_AGCWD(input_dir, output_dir):
    img_paths = glob.glob(input_dir+'*')
    total_count= 0
    dimmed_count= 0
    brighted_count = 0
    for path in img_paths:
        total_count +=1
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Convert to BGR
        # img = cv2.resize(img, (224,224)) # Resize to 224x224
        name = path.split('\\')[-1].split('.')[0]
        
        # Extract intensity component of the image
        YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        Y = YCrCb[:,:,0]
        # Determine whether image is bright or dimmed
        threshold = 0.3
        exp_in = 112 # Expected global average intensity 
        M,N = img.shape[:2]
        mean_in = np.sum(Y/(M*N)) 
        t = (mean_in - exp_in)/ exp_in
        
        # Process image for gamma correction
        img_output = None
        if t < -threshold: # Dimmed Image
            dimmed_count += 1
            # print (name + ": Dimmed")
            result = process_dimmed(Y)
            YCrCb[:,:,0] = result
            img_output = cv2.cvtColor(YCrCb,cv2.COLOR_YCrCb2BGR)
        elif t > threshold:
            brighted_count += 1
            # print (name + ":Bright") # Bright Image
            result = process_bright(Y)
            YCrCb[:,:,0] = result
            img_output = cv2.cvtColor(YCrCb,cv2.COLOR_YCrCb2BGR)
        else:
            # print('None')
            img_output = img
            img_output = cv2.cvtColor(img_output, cv2.COLOR_BGR2GRAY)
        
        cv2.imwrite(output_dir+name+'.png', img_output)
        # print(total_count, output_dir)
    print("Total number of processed images: {}".format(total_count))
    print("Total number of processed dimmed images: {}".format(dimmed_count))
    print("Total number of processed brighted images: {}".format(brighted_count))
    return

if __name__ == '__main__':
    '''
    Set the main path for both original and processed datasets : ./dataset 
    Input and output paths must be set for each class
    '''
    input_dir = 'C:\\Users\\marco\\Desktop\\Local_Documents\\data\\COVIDx-splitted-resized-112-process-enanch-augm'
    # output_dir = 'C:\\Users\\marco\\Desktop\\Local_Documents\\data\\COVIDx-splitted-resized-112-process-enanch-augm'


    # COVID-19 images
    input_dir_covid = os.path.join(input_dir,'test\\COVID-19\\')
    # output_dir_covid = os.path.join(output_dir, 'test\\COVID-19\\')
    apply_AGCWD(input_dir=input_dir_covid, output_dir=input_dir_covid)

    # # Normal images
    input_dir_normal = os.path.join(input_dir, 'test\\normal\\')
    # output_dir_normal = os.path.join(output_dir, 'test\\normal\\')
    apply_AGCWD(input_dir=input_dir_normal, output_dir=input_dir_normal)

    # Pneumonia images
    input_dir_pneumonia = os.path.join(input_dir, 'test\\pneumonia\\')
    # output_dir_pneumonia = os.path.join(input_dir, 'test\\pneumonia\\')
    apply_AGCWD(input_dir=input_dir_pneumonia, output_dir=input_dir_pneumonia)