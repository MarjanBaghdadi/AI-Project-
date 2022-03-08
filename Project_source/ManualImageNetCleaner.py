"""
    This script is intended to clean the files that were downloaded using the
    'imageNetDownloader.py' script. These files include broken urls, files
    without a specific format and so on.

    The cleaning includes both removing the files from the ImageNet dataset folder
    and also removing the corresponding entries in the ImageNet Reference csv file,
    and finally saving the cleaned csv file as reference text file.
"""
import pandas as pd
import os


root_dir = "Dataset/ImageNet_dataset/"
csv_ref_file_name = "ImageNet_References.csv"
manual_selected_files = 'Manual_Removal.txt'
dataset_dir = root_dir + 'Downloaded_images/'

image_ref = pd.read_csv(root_dir + csv_ref_file_name)

with open(root_dir + manual_selected_files, encoding='utf-8', errors='ignore') as f:
    file_names = f.readlines()

    for file_name in file_names:
        file_name = file_name.replace("\n", "")

        # Removing the file from dataset directory
        if os.path.exists(dataset_dir + file_name):
            os.remove(dataset_dir + file_name)

        # Removing from the ref File
        file_index = image_ref[image_ref['Image_Name'] == file_name].index.values
        image_ref.drop(file_index, inplace=True)

    # Updating the csv reference file and also saving it as a text file
    image_ref.reset_index(drop=True, inplace=True)
    image_ref.to_csv(root_dir + csv_ref_file_name, index=False)

    image_ref.to_csv(root_dir + "ImageNet_References.txt", header=None, index=None, sep=' ', mode='a')
