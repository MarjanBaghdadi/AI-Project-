import torch
import dataLoader as dL
from CNN_Model import FaceDetector
import pandas as pd
import os

from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


if __name__ == "__main__":
    csv_file = "Dataset/Train/train_dataset_info.csv"
    root_dir = ""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    #####################################################################
    #           Creating the train and test datasets                    #
    #####################################################################

    file_info = pd.read_csv(csv_file)

    train_set = dL.MaskedDataset(file_info, root_dir, transform=transforms.Compose([dL.Rescale(128), dL.ToTensor()]))
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)

    faceDetectorModel = FaceDetector()
    faceDetectorModel.to(device)

    #####################################################################
    #             Configuring the Loss and Optimizer                    #
    #####################################################################
    maskNum = file_info[file_info['mask'] == 1].shape[0]
    nonMaskNum = file_info[file_info['mask'] == 0].shape[0]
    notPersonNum = file_info[file_info['mask'] == 2].shape[0]
    nSamples = [nonMaskNum, maskNum, notPersonNum]
    normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(normedWeights)).to(device)

    optimizer = torch.optim.Adam(faceDetectorModel.parameters(), lr=0.0001)

    #####################################################################
    #                       Training the network                        #
    #####################################################################
    epoch_num = 10
    losses = []
    batch_num_per_epoch = 0
    for epoch in range(epoch_num):  # loop over the dataset multiple times

        if epoch > 0:
            model_name_epoch = 'trainedModels/faceMaskNet_3_class_final_epoch_' + str(epoch)
            torch.save(faceDetectorModel.state_dict(), model_name_epoch + '.pth')

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
            losses.append(loss)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 250 == 249:    # print every 250 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 250))
                running_loss = 0.0

        if batch_num_per_epoch == 0:
            batch_num_per_epoch = i + 1

    #####################################################################
    #                       Saving the trained model                    #
    #####################################################################
    path = 'trainedModels/'
    model_name = 'faceMaskNet_3_class_final.pth'

    if not os.path.isdir(path):
        os.mkdir(path)

    torch.save(faceDetectorModel.state_dict(), os.path.join(path, model_name))
    torch.cuda.empty_cache()

    #####################################################################
    #                       Plotting loss vs epoch                      #
    #####################################################################
    epoch_mean_loss = []
    for i in range(epoch_num):
        epoch_loss = 0
        for j in range(batch_num_per_epoch):
            epoch_loss += losses[j + (i * batch_num_per_epoch)].item()
        epoch_mean_loss.append(epoch_loss / batch_num_per_epoch)

    plt.plot([x + 1 for x in range(len(epoch_mean_loss))], epoch_mean_loss)
    plt.title("Loss vs epoch")
    plt.xlabel("epoch number")
    plt.ylabel("loss value")
