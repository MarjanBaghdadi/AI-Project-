"""
    This script reads the data in the train and test folders and creates a csv file for
    each folder containing the image addresses and their corresponding class labels.
"""
import pandas as pd
from pathlib import Path


def create_dataset_info_csv_file(data_directory, save_file_name):
    """

    :param data_directory: The directory that contains the dataset images. It is assumed that the given directory
    has an specified structure with the three solder of 'WithMask', 'WithoutMask', and 'NotPerson'.

    :param save_file_name: the full name (with dir) of the csv file in which the extracting info will be saved.

    :return: A csv file containing the address and label of the dataset images.
    """
    data_dir = Path(data_directory)
    df_object = pd.DataFrame()

    for folder in list(data_dir.iterdir()):
        image_label = 0
        if folder.name == 'WithMask':
            image_label = 1
        elif folder.name == 'NotPerson':
            image_label = 2

        for image_path in folder.iterdir():
            df_object = df_object.append({
                "image": str(image_path),
                "mask": image_label
            }, ignore_index=True)

    print("saving Dataframe to: ", save_file_name)
    df_object.to_csv(save_file_name)


if __name__ == "__main__":
    train_dir = 'Dataset/Train/'
    test_dir = 'Dataset/Test/'

    create_dataset_info_csv_file(train_dir, train_dir + 'train_dataset_info.csv')
    create_dataset_info_csv_file(test_dir, test_dir + 'test_dataset_info.csv')
