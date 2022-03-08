"""
    This file is intended to randomly split the ImageNet and RMFD dataset images based on
    their classes into train and test splits, and move them into the final dataset folder.

    Author: Ali Ghelmani,       Date: 12/6/2020
"""
from shutil import copy2
from sklearn.model_selection import train_test_split
import pandas as pd
import os


def copy_without_replacement(input_file, dest_dir):
    """

    :param input_file: The name (with dir) of the file to be copied
    :param dest_dir: The copying destination!
    :return: void

    As the name implies this function checks in the destination folder before
    copying to see if a name conflict exists. If detected modifies the copied
    name by adding '_{number}' to the end of the file.
    """
    dest_fileName = dest_dir + os.path.basename(input_file)
    present_counter = 0

    while os.path.exists(dest_fileName):    # recursively check for existing file names
        name, file_extension = os.path.splitext(dest_fileName)
        present_counter += 1

        if present_counter > 1:     # remove the previous instance of '_{num}' before increasing num and adding anew
            base_split = name.split('_')
            name = '_'.join(base_split[:-1])

        dest_fileName = name + f'_{present_counter}' + file_extension

    copy2(input_file, dest_fileName)


def copyImageData_RMFD(input_dataframe, dest_dir):
    """

    :param input_dataframe: A dataframe containing the RMFD dataset image addresses and their corresponding labels
    :param dest_dir: The destination to copy the RMFD images into
    :return: void

    A simple function that changes the copying destination of the RMFD images based on their mask label.
    """
    mask_dir = dest_dir + 'WithMask/'
    without_mask_dir = dest_dir + 'WithoutMask/'

    for index, image in input_dataframe.iterrows():
        if image['mask'] == 1:
            copy_without_replacement(image['image'], mask_dir)
        elif image['mask'] == 0:
            copy_without_replacement(image['image'], without_mask_dir)


def copyImageData_ImageNet(input_image_list, src_dir, dest_dir):
    """

    :param input_image_list: List of image filenames of the ImageNet dataset to be copied.
    :param src_dir: The directory containing the ImageNet images
    :param dest_dir: The destination for images to be copied into.
    :return: void

    This function checks for the existence of 'NotPerson' folder in the destination directory
    and will create one if necessary, after that simply passes the images to the 'copy_without_replacement'
    function for copying.
    """
    not_person_dir = dest_dir + 'NotPerson/'
    if not os.path.exists(not_person_dir):
        os.mkdir(not_person_dir)

    for image in input_image_list:
        copy_without_replacement(src_dir + image, not_person_dir)


if __name__ == "__main__":
    split_ratio = 0.85
    save_dir = 'Dataset/'

    #########################################
    #               RMFD case               #
    #########################################
    root_dir = "Dataset/self-built-masked-face-recognition-dataset/"
    csv_fileName = "dataset_ver2_short_version.csv"

    # Dividing the data into train and set sections
    file_info = pd.read_csv(root_dir + csv_fileName)
    train_data, test_data = train_test_split(file_info, train_size=split_ratio, random_state=0,
                                             stratify=file_info['mask'])
    print(f'RMFD train size: {len(train_data)}, test size: {len(test_data)}')

    # Removing possible CIFAR10 files from train and test data
    train_data = train_data[train_data['mask'] < 2]
    test_data = test_data[test_data['mask'] < 2]

    # Copying the files!
    copyImageData_RMFD(train_data, save_dir + 'Train/')
    copyImageData_RMFD(test_data, save_dir + 'Test/')

    #########################################
    #             ImageNet case             #
    #########################################
    imageNet_dir = 'Dataset/ImageNet_dataset/'
    image_list = os.listdir(imageNet_dir + 'Downloaded_images')

    # Dividing the data into train and set sections
    train_data, test_data = train_test_split(image_list, train_size=split_ratio, random_state=0)

    # Copying the files!
    copyImageData_ImageNet(train_data, imageNet_dir + 'Downloaded_images/', save_dir + 'Train/')
    copyImageData_ImageNet(test_data, imageNet_dir + 'Downloaded_images/', save_dir + 'Test/')

    print(f'ImageNet train size: {len(train_data)}, test size: {len(test_data)}')
