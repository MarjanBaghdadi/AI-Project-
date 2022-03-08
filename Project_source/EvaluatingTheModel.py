from CNN_Model import FaceDetector
import dataLoader as dL
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.metrics import f1_score, recall_score
from torchvision import transforms
import pandas as pd
import torch
import os


def evaluateTheTrainedModel(input_model_name, test_dataLoader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    testNet = FaceDetector()
    testNet.load_state_dict(torch.load(input_model_name))
    testNet.to(device)

    with torch.no_grad():
        pred = torch.tensor([]).to(device)
        labels = torch.tensor([]).to(device)

        for data in test_dataLoader:
            inputs = data['image'].to(device)
            data_labels = data['label'].to(device)

            outputs = testNet(inputs.float())
            _, predicted = torch.max(outputs, 1)

            pred = torch.cat((pred, predicted), dim=0)
            labels = torch.cat((labels, data_labels.squeeze()), dim=0)

    labels = labels.cpu()
    pred = pred.cpu()
    torch.cuda.empty_cache()

    return labels, pred


if __name__ == "__main__":

    csv_file = "Dataset/Test/test_dataset_info.csv"
    root_dir = ""

    file_info = pd.read_csv(csv_file)

    test_set = dL.MaskedDataset(file_info, root_dir, transform=transforms.Compose([dL.Rescale(128), dL.ToTensor()]))
    test_loader = DataLoader(test_set, batch_size=32, shuffle=True, num_workers=4, drop_last=True)

    #####################################################################
    #                          Testing the model                        #
    #####################################################################
    path = 'trainedModels/'
    model_name = 'faceMaskNet_3_class_final.pth'

    all_labels, all_preds = evaluateTheTrainedModel(os.path.join(path, model_name), test_loader)

    correct = (all_preds == all_labels).sum().item()

    print('Network Accuracy: %.4f' % (100 * correct / all_labels.shape[0]))

    print('Precision: ', precision_score(all_labels, all_preds, average=None))

    print('Recall: ', recall_score(all_labels, all_preds, average=None))

    print('F1-Measure: ', f1_score(all_labels, all_preds, average=None))

    print('confusion matrix: \n', confusion_matrix(all_labels, all_preds))
