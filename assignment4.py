# get the datasets
import gdown

url = "https://drive.google.com/drive/folders/1ZxOE5BPcmW5RQUK_XGe6cHXplKBvvxgf?usp=share_link"
gdown.download_folder(url, quiet=True, use_cookies=False)
import tarfile
file = tarfile.open('skeletonizable_images/outdoors_climbing_00.tar.xz')
file.extractall('data')
file.close()
file = tarfile.open('skeletonizable_images/outdoors_climbing_01.tar.xz')
file.extractall('data')
file.close()

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.io import read_image
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
from os.path import exists
import torch.optim as optim
import torch.nn as nn

########### Block 1
# create a data loader for the dataset
class KeypointDataset(Dataset):
    def __init__(self, folder_path, folder_path_temp, sequence_length, device='gpu'):
        self.folder_path = folder_path
        self.folder_path_temp = folder_path_temp
        self.files = os.listdir(folder_path)
        self.files.sort()
        self.sequence_length = sequence_length
        self.image_width = 1080
        self.image_height = 1920
        self.device = device

        # set the keypoint extractor
        weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
        self.transforms = weights.transforms()
        self.keypoint_predition_model = keypointrcnn_resnet50_fpn(weights=weights, progress=False)
        self.keypoint_predition_model = self.keypoint_predition_model.eval()
        self.keypoint_predition_model = self.keypoint_predition_model.to(device)

    def __len__(self):
        return len(self.files) - (self.sequence_length+1) # each sequence needs the prediction as a ground truth

    def __getitem__(self, idx):
        # load the images as a batch
        keypoints = torch.empty(self.sequence_length+1, 17, 3)
        for idx_offset in range(self.sequence_length+1):
            file_name = self.files[idx+idx_offset]
            keypoints_path = os.path.join(self.folder_path_temp, file_name[:-4] + '.pt')
            keypoints[idx_offset] = torch.load(keypoints_path, map_location=torch.device(self.device))
        # perform a shift between the features and the prediction
        X_train = keypoints[0:self.sequence_length, :, :]
        y_train = keypoints[1:self.sequence_length+1, :, :]
        return X_train, y_train

    def preprocessData(self):
        for idx in range(len(self.files)):
            file_name = self.files[idx]
            image_path = os.path.join(self.folder_path, file_name)
            image = read_image(image_path)

            # preprocess the image (hint: use self.transforms)
            image = self.transforms(image)


            # predict the keypoints (hint: use self.keypoint_predition_model)
            # read this discussion if you run out of memory: https://discuss.pytorch.org/t/out-of-memory-error-during-evaluation-but-training-works-fine/12274
            with torch.no_grad():
              outputs = self.keypoint_predition_model([image])

            keypoints = outputs[0]['keypoints'][0] # first person in the frame
            keypoint_path = os.path.join(self.folder_path_temp, file_name[:-4] + '.pt')
            torch.save(keypoints, keypoint_path)

    def returnImage(self, idx):
        file_name = self.files[idx]
        image_path = os.path.join(self.folder_path, file_name)
        image = read_image(image_path)
        return image

    def returnKeypoints(self, idx):
        file_name = self.files[idx]
        keypoints_path = os.path.join(self.folder_path_temp, file_name[:-4] + '.pt')
        keypoints = torch.load(keypoints_path, map_location=torch.device(self.device))
        return keypoints

########### Block 2
sequence_length = 10
split_ratio = 0.67
batch_size = 10
device = 'cpu'
#device = 'cuda:0' # set the keypoint extractor on the GPU

folder_path = "data/outdoors_climbing_00"
folder_path_temp = "data/outdoors_climbing_00_keypoints"
os.mkdir(folder_path_temp)
dataset_training = KeypointDataset(folder_path, folder_path_temp, sequence_length, device=device)
if not exists("data/outdoors_climbing_00_keypoints/image_00000.pt"):
    dataset_training.preprocessData()
data_loader_training = torch.utils.data.DataLoader(dataset_training, batch_size=batch_size, shuffle=False)

folder_path = "data/outdoors_climbing_01"
folder_path_temp = "data/outdoors_climbing_01_keypoints"
os.mkdir(folder_path_temp)
dataset_testing = KeypointDataset(folder_path, folder_path_temp, sequence_length, device=device)
if not exists("data/outdoors_climbing_01_keypoints/image_00000.pt"):
    dataset_testing.preprocessData()
data_loader_testing = torch.utils.data.DataLoader(dataset_testing, batch_size=batch_size, shuffle=False)


########### Block 3
class LSTM(nn.Module):
    def __init__(self, LSTM_input_size):
        super().__init__()
        # define the LSTM model
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)


    def forward(self, x):
        # define the forward pass
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

LSTM_input_size = 1*2 # 1 keypoints with (x,y) coordinates

LSTM_model = LSTM(LSTM_input_size)
optimizer = optim.Adam(LSTM_model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

########### Block 4
#i, batch = next(enumerate(data_loader))
epochs = 40
training_loss = []
testing_loss = []
for epoch in range(epochs):
    for X_batch, y_batch in data_loader_training:
        print(X_batch.size)
        LSTM_model.train()
        # Let's pick the first keypoint and its (x,y) coordinates
        # Hint: X_batch is organized as (batch_size, sequence_length, number_of_keypoints, [x,y,z])

        # pass through the LSTM model
        y_pred_tr = LSTM_model(X_batch[:,0,:,:])

        # loss function (remember the LSTM output and the y_batch must have the same shape)
        loss_tr = loss_fn(y_pred_tr, y_batch[0,:])

        # backpropagation
        loss_tr.backward()
        optimizer.step()

    # testing
    if epoch % 10 == 0:
        with torch.no_grad():

            # compute the loss function for the training set
            total_loss = 0
            number_of_batches = 0
            for X_batch, y_batch in data_loader_training:
                LSTM_model.eval()

                # Let's pick the first keypoint
                y_pred = LSTM_model(X_train)


                # pass through the LSTM model
                train_RMSE = np.sqrt(loss_fn(y_pred, y_train))

                # loss function
                train_RMSE.append(train_RMSE)

                total_loss += train_RMSE.item() # loss.item() returns the loss value as a float (free of the gradient)
                number_of_batches += 1

            training_loss.append(total_loss/number_of_batches)

            # compute the loss function for the testing set
            total_loss = 0
            number_of_batches = 0
            for X_batch, y_batch in data_loader_testing:
                LSTM_model.eval()

                # Let's pick the first keypoint
                y_pred_te = LSTM_model(X_test)

                # pass through the LSTM model
                test_RMSE = np.sqrt(loss_fn(y_pred_te, y_test))

                # loss function
                test_RMSE.append(test_RMSE)

                total_loss += test_RMSE.item()
                number_of_batches += 1

            testing_loss.append(total_loss/number_of_batches)

        # print the loss
        print('Epoch: ', epoch, '\tTraining loss: ', '% 6.2f' % training_loss[-1], '\tTesting loss: ', '% 6.2f' % testing_loss[-1])
