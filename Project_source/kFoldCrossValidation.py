import torch
import dataLoader as dL
from CNN_Model import FaceDetector
import pandas as pd
import numpy as np
import pickle
import os

from torchvision import transforms
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix, precision_score
from sklearn.metrics import f1_score, recall_score


def train_model_on_fold_data(fold_data, saved_model_name, leftOut_fold_number, leftOut_data):
    """

    :param fold_data: The (k - 1) sections of the input data that are used for training
    :param saved_model_name: The desired name for saving the model. Preferably containing some clues
    about the structure of the CV mode
    :param leftOut_fold_number: a number between 0 and k - 1, indicating the leftOut section of the data
    :param leftOut_data: The left out section of the training data that is used for testing
    :return: The true label and prediction results on the testing fold to be later used for various
    metric calculations.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    root_dir = ""
    input_image_size = 128

    train_set = dL.MaskedDataset(fold_data, root_dir, transform=transforms.Compose([dL.Rescale(input_image_size), dL.ToTensor()]))
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)

    faceDetectorModel = FaceDetector()
    faceDetectorModel.to(device)

    #####################################################################
    #             Configuring the Loss and Optimizer                    #
    #####################################################################
    maskNum = fold_data[fold_data['mask'] == 1].shape[0]
    nonMaskNum = fold_data[fold_data['mask'] == 0].shape[0]
    notPersonNum = fold_data[fold_data['mask'] == 2].shape[0]
    nSamples = [nonMaskNum, maskNum, notPersonNum]
    normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(normedWeights)).to(device)

    optimizer = torch.optim.Adam(faceDetectorModel.parameters(), lr=0.0001)

    #####################################################################
    #                       Training the network                        #
    #####################################################################
    epoch_num = 10
    training_losses = []
    for epoch in range(epoch_num):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data['image'].to(device)
            labels = data['label'].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = faceDetectorModel(inputs.float())
            loss = criterion(outputs, labels.long().view(-1))
            training_losses.append(loss)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 250 == 249:  # print every 250 mini-batches
                print('LeftOut fold number: %d, [%d, %5d] loss: %.3f' %
                      (leftOut_fold_number, epoch + 1, i + 1, running_loss / 250))
                running_loss = 0.0

    #####################################################################
    #         Saving the trained model and the training losses          #
    #####################################################################
    path = f'trainedModels/{saved_model_name}/'
    model_name = f'{saved_model_name}_{leftOut_fold_number}.pth'

    if not os.path.isdir(path):
        os.mkdir(path)

    torch.save(faceDetectorModel.state_dict(), os.path.join(path, model_name))
    with open(os.path.join(path, model_name).replace('pth', 'pkl'), 'wb') as f:
        pickle.dump(training_losses, f)

    #####################################################################
    #             Testing the model on the left out fold                #
    #####################################################################
    test_set = dL.MaskedDataset(leftOut_data, root_dir, transform=transforms.Compose([dL.Rescale(input_image_size), dL.ToTensor()]))
    test_dataLoader = DataLoader(test_set, batch_size=32, shuffle=True, num_workers=4, drop_last=True)

    with torch.no_grad():
        pred = torch.tensor([]).to(device)
        labels = torch.tensor([]).to(device)

        for data in test_dataLoader:
            inputs = data['image'].to(device)
            data_labels = data['label'].to(device)

            outputs = faceDetectorModel(inputs.float())
            _, predicted = torch.max(outputs, 1)

            pred = torch.cat((pred, predicted), dim=0)
            labels = torch.cat((labels, data_labels.squeeze()), dim=0)

    labels = labels.cpu()
    pred = pred.cpu()
    torch.cuda.empty_cache()

    return labels, pred


def crossValidationLossResults(test_results, fold_number):
    """

    :param test_results: A list containing the label and prediction results for various test folds
    :param fold_number: The number of folds considered
    :return: The mean accuracy, precision, recall, f1_score, and confusion_matrix over folds
    """
    accuracy = 0
    precision = np.array([])
    recall = np.array([])
    f_1_score = np.array([])
    confusion_mat = np.array([])

    for index, data in enumerate(test_results):
        label = data[0]
        pred = data[1]
        correct = (pred == label).sum().item()
        accuracy += (100 * correct / label.shape[0])

        if index == 0:
            precision = precision_score(label, pred, average=None)
            recall = recall_score(label, pred, average=None)
            f_1_score = f1_score(label, pred, average=None)
            confusion_mat = confusion_matrix(label, pred)
        else:
            precision += precision_score(label, pred, average=None)
            recall += recall_score(label, pred, average=None)
            f_1_score += f1_score(label, pred, average=None)
            confusion_mat += confusion_matrix(label, pred)

    print(f'Accuracy: {accuracy / fold_number}')
    print(f'Precision: {precision / fold_number}')
    print(f'Recall: {recall / fold_number}')
    print(f'F1_score: {f_1_score / fold_number}')
    print(f'Confusion Matrix: {confusion_mat / fold_number}')


if __name__ == "__main__":
    csv_file = "Dataset/Train/train_dataset_info.csv"
    file_info = pd.read_csv(csv_file)
    fold_numbers = 5
    save_name = 'faceMaskNet_3_class_Net7'

    rand_perm = np.random.permutation(len(file_info))
    fold_size = int(np.floor(len(file_info) / fold_numbers))
    fold_test_results = []

    for fold_index in range(fold_numbers):
        data_index = []
        leftOut_index = []

        if fold_index == 0:
            data_index = rand_perm[fold_size:]
            leftOut_index = rand_perm[:fold_size]

        elif fold_index < fold_numbers - 1:
            data_index = rand_perm[:(fold_index * fold_size)]
            data_index = np.append(data_index, rand_perm[((fold_index + 1) * fold_size):])

            leftOut_index = rand_perm[(fold_index * fold_size):((fold_index + 1) * fold_size)]
        else:
            data_index = rand_perm[:-fold_size]
            leftOut_index = rand_perm[-fold_size:]

        fold_data = file_info.loc[data_index, :]
        leftOut_data = file_info.loc[leftOut_index, :]

        fold_test_results.append(train_model_on_fold_data(fold_data, save_name,
                                                    fold_index, leftOut_data))
    # Saving the model results
    fold_loss_name = f'trainedModels/{save_name}/fold_losses.pkl'
    with open(fold_loss_name, 'wb') as f:
        pickle.dump(fold_test_results, f)

    # now perform the mean of the results
    crossValidationLossResults(fold_test_results, fold_numbers)
