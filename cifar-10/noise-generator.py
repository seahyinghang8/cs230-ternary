""" 
NOISE GENERATOR PROGRAM
- Add various levels of noise to the test_batch and save it to the destination folder
- gaussian, salt and pepper, poisson, speckles
"""

import os
import shutil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import filters
from scipy import misc
import pickle

# OVERALL VARIABLES FOR THE PROGRAM
test_batch = "cifar.python/cifar-10-batches-py/test_batch"
meta = "cifar.python/cifar-10-batches-py/batches.meta"
destination = "images/"
dataset_max = 100

#unpickle
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Source of the code is based on an excelent piece code from stackoverflow
# http://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
def noise_generator (noise_type, image, 
                     #for gaussian
                     mean = 0, var = 0.01, 
                     #for Salt/Pepper
                     s_vs_p = 0.5, amount = 0.004
                    ):
    """
    Generate noise to a given Image based on required noise type
    
    Input parameters:
        image: ndarray (input image data. It will be converted to float)
        
        noise_type: string
            'gauss'        Gaussian-distrituion based noise
            'poission'     Poission-distribution based noise
            's&p'          Salt and Pepper noise, 0 or 1
            'speckle'      Multiplicative noise using out = image + n*image
                           where n is uniform noise with specified mean & variance
    """
    row, col, ch = image.shape
    if noise_type == "gauss":       
        sigma = var**0.5
        gauss = np.array(image.shape)
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy.astype('uint8')
    elif noise_type == "s&p":
        out = image
        # Generate Salt '1' noise
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
        out[coords] = 255
        # Generate Pepper '0' noise
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
        out[coords] = 0
        return out
    elif noise_type == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_type =="speckle":
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)        
        noisy = image + image * gauss
        return noisy
    else:
        return image


def process_image(index, dataset, label_names, directory_name):
    # PROCESS IMAGE
    raw_arr = dataset[b'data'][index,:]
    normal_image = raw_arr.reshape(3,32,32).transpose([1, 2, 0])
    # GET SAVE PATH
    class_index = dataset[b'labels'][index]
    label = label_names[class_index].decode("utf-8")
    save_path = os.path.join(directory_name, label, str(index) + ".jpg")
    return normal_image, save_path

def make_label_directory(directory, label_names):
    # Delete existing data and create the directory
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

    # Loop through all the labels and create the directory
    for label in label_names:
        label_str = label.decode("utf-8")
        label_path = os.path.join(directory, label_str)
        os.makedirs(label_path)

def main():
    dataset = unpickle(test_batch)
    label_names = unpickle(meta)[b'label_names']
    dataset_size = len(dataset[b'data'])
    dataset_size = dataset_max if dataset_size > dataset_max else dataset_size

    print("Generating a data of %d images" %(dataset_size))
    
    print("Generating clean images (No noise added)")
    
    clean_directory = os.path.join(destination, "clean")
    make_label_directory(clean_directory, label_names)
    
    for image_counter in range(0, dataset_size):
        normal_image, save_path = process_image(image_counter, dataset, label_names, clean_directory)
        misc.imsave(save_path, normal_image)

    print("Generating salt and pepper noise")
    
    for noise_level in range(1, 10):
        print("Adding noise level: " + str(noise_level))

        salt_and_pepper_directory = os.path.join(destination, "salt-and-pepper-noise-" + str(noise_level))
        make_label_directory(salt_and_pepper_directory, label_names)

        for image_counter in range(0, dataset_size):
            normal_image, save_path = process_image(image_counter, dataset, label_names, salt_and_pepper_directory)
            noised_image = noise_generator(noise_type='s&p', image=normal_image, amount=0.001*noise_level)
            misc.imsave(save_path, noised_image)

    print("Generating gaussian noise")

    for noise_level in range(1, 2):
        print("Adding noise level: " + str(noise_level))

        gauss_directory = os.path.join(destination, "gaussian-noise-" + str(noise_level))
        make_label_directory(gauss_directory, label_names)

        for image_counter in range(0, dataset_size):
            normal_image, save_path = process_image(image_counter, dataset, label_names, gauss_directory)
            # NEED TO COME UP WITH A BETTER GAUSSIAN VARIANCE SCALE
            variance = 0.00000000000000001 * (10 ** noise_level)
            noised_image = noise_generator(noise_type='gauss', image=normal_image, var=0)
            misc.imsave(save_path, noised_image)

    print("Generating poisson noise")
    
    poisson_directory = os.path.join(destination, "poisson")
    make_label_directory(poisson_directory, label_names)
    
    for image_counter in range(0, dataset_size):
        normal_image, save_path = process_image(image_counter, dataset, label_names, poisson_directory)
        noised_image = noise_generator(noise_type='poisson', image=normal_image)
        misc.imsave(save_path, noised_image)

    print("Generating speckle noise")
    
    speckle_directory = os.path.join(destination, "speckle")
    make_label_directory(speckle_directory, label_names)
    
    for image_counter in range(0, dataset_size):
        normal_image, save_path = process_image(image_counter, dataset, label_names, speckle_directory)
        noised_image = noise_generator(noise_type='speckle', image=normal_image)
        misc.imsave(save_path, noised_image)

# run main
main()