# AI-Project-
The list of files submitted:


Report.pdf (The project report)

trainedModels - faceMaskNet_3_class_final.pth (The Final Trained Model)

CNN_Model.py (Defining the CNN structure)

dataLoader.py (Loading different parts of data, and processing them in different classes for multiple purposes in other files like RMFD_fileNameModifier.py, main.py , and EvaluatingTheModel.py)

EvaluatingTheModel.py (Evaluating and testing the model, and computing the performance)

main.py (The main code for training the finally selected model)

prep.py (Prepping the RMFD images as part of the preprocessing)

RMFD_fileNameModifier.py (Editing file names to make them readable for the rest of the code)

ImageNetDownloader.py (used to download the the required number of samples from each of the selected ImageNet classes using the URL text files)

ManualImageNetCleaner.py (manual removal of corrupt images received by the above module.)

ImageNet_RMFD_Split.py (dividing and adding the RMFD and ImageNet images to the kaggle dataset images)

prepVersion2.py (from the final images in the train and test dataset creates a path and label csv file used by the dataLoader)

kFoldCrossValidation.py (implementing the 10-fold croos validation on the traiing data and calculating the average of various measure calculated on various iterations)

Dataset - ImageNet_dataset - ImageNet_References.txt (contains the links of all of the downloaded ImageNet images) 

Dataset - ImageNet_dataset - Selected_Class_URLS (folder containing text files of the selected classes from the ImageNet dataset)

Dataset - ImageNet_dataset (sample images from each class in our dataset)

How to run the code:


ImageNetDownloader.py – Downloading images from the ImageNet dataset

ManualImageNetCleaner.py – cleaning the mistaken downloaded images

ImageNet_RMFD_Split.py - move and merge the RMFD and ImageNet data with the kaggle one

prepVersion2.py - creating the info csv files that are later used in the dataloader

kFoldCrossValidation.py the main file that performs the k-fold croos validation

fileNameModifier.py – Omitting the Chinese characters from the .csv file and gaining the final form of it (The one that can be found in Dataset_csv_files.zip)

main.py – Training the model from the selected architecture

EvaluatingTheModel.py – Testing the model and getting the performance measures as an output printed in Console
