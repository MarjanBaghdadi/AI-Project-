"""
    This script drops some of the image file names in the RMFD dataset that have chinese
    characters in them and might results in the FileNotFoundError when loading the dataset.
"""
from dataLoader import MaskedDataset
import pandas as pd

csv_file = "Dataset/self-built-masked-face-recognition-dataset/dataset_ver2_short_version.csv"
root_dir = ""

data_loader = MaskedDataset(pd.read_csv(csv_file), root_dir)

# Finding the problematic files
problem_names = []
for image_idx in range(data_loader.__len__()):
    try:
        sample = data_loader.__getitem__(image_idx)
    except FileNotFoundError:
        problem_names.append(image_idx)

print(len(problem_names))
print(problem_names)

# Removing the files with unsupported file names from the dataset
file_list = pd.read_csv(csv_file)
file_list.set_index('Unnamed: 0', inplace=True)

for name_idx in reversed(problem_names):
    file_list.drop([name_idx], inplace=True)

file_list.reset_index(drop=True, inplace=True)
file_list.to_csv(csv_file)
