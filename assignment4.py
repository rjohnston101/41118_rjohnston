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
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

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
#os.mkdir(folder_path_temp)
dataset_training = KeypointDataset(folder_path, folder_path_temp, sequence_length, device=device)
if not exists("data/outdoors_climbing_00_keypoints/image_00000.pt"):
    dataset_training.preprocessData()
data_loader_training = torch.utils.data.DataLoader(dataset_training, batch_size=batch_size, shuffle=False)

folder_path = "data/outdoors_climbing_01"
folder_path_temp = "data/outdoors_climbing_01_keypoints"
#os.mkdir(folder_path_temp)
dataset_testing = KeypointDataset(folder_path, folder_path_temp, sequence_length, device=device)
if not exists("data/outdoors_climbing_01_keypoints/image_00000.pt"):
    dataset_testing.preprocessData()
data_loader_testing = torch.utils.data.DataLoader(dataset_testing, batch_size=batch_size, shuffle=False)


########### Block 3
class LSTM(nn.Module):
    def __init__(self, LSTM_input_size):
        super().__init__()
        # define the LSTM model
        self.lstm = nn.LSTM(input_size=LSTM_input_size, hidden_size=64, num_layers=2, batch_first=True)
        self.linear = nn.Linear(64, 3) #  Account for 3D keypoints


    def forward(self, x):
        # define the forward pass
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

LSTM_input_size = 3 # 3D keypoints

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
        LSTM_model.train()
        optimizer.zero_grad()
        # Let's pick the first keypoint and its (x,y) coordinates
        # Hint: X_batch is organized as (batch_size, sequence_length, number_of_keypoints, [x,y,z])
        
        # pass through the LSTM model
        X_batch = X_batch.view(-1, X_batch.size(2), X_batch.size(3)) #  reshape x_batch to suit LSTM_model
        y_pred_tr = LSTM_model(X_batch)
        y_pred_tr = y_pred_tr.view(y_batch.size(0), y_batch.size(1), y_batch.size(2), -1)

        # loss function (remember the LSTM output and the y_batch must have the same shape)
        loss_tr = loss_fn(y_pred_tr, y_batch)

        # backpropagation
        loss_tr.backward()
        optimizer.step()

    # testing
    if epoch % 10 == 0:
        with torch.no_grad():
            total_loss = 0
            number_of_batches = 0
            for X_batch, y_batch in data_loader_training:
                LSTM_model.eval()

                #train_RMSE = np.sqrt(loss_fn(y_pred, y_train))

                #train_RMSE.append(train_RMSE)

                #total_loss += train_RMSE.item() # loss.item() returns the loss value as a float (free of the gradient)

                X_batch = X_batch.view(-1, X_batch.size(2), X_batch.size(3))
                y_pred_tr = LSTM_model(X_batch)
                y_pred_tr = y_pred_tr.view(y_batch.size(0), y_batch.size(1), y_batch.size(2), -1)
                loss_tr = loss_fn(y_pred_tr, y_batch)

                total_loss += loss_tr.item() # loss.item() returns the loss value as a float (free of the gradient)
                number_of_batches += 1

            training_loss.append(total_loss/number_of_batches)

            # compute the loss function for the testing set
            total_loss = 0
            number_of_batches = 0
            for X_batch, y_batch in data_loader_testing:
                LSTM_model.eval()
                X_batch = X_batch.view(-1, X_batch.size(2), X_batch.size(3))
                y_pred_tr = LSTM_model(X_batch)
                y_pred_tr = y_pred_tr.view(y_batch.size(0), y_batch.size(1), y_batch.size(2), -1)
                loss_tr = loss_fn(y_pred_tr, y_batch)

                total_loss += loss_tr.item() # loss.item() returns the loss value as a float (free of the gradient)
                number_of_batches += 1

            testing_loss.append(total_loss/number_of_batches)

        # print the loss
        print('Epoch: ', epoch, '\tTraining loss: ', '% 6.2f' % training_loss[-1], '\tTesting loss: ', '% 6.2f' % testing_loss[-1])

####### Block 5
plt.plot(training_loss, label='Training Loss')
plt.plot(testing_loss, label='Testing Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Testing Loss')
plt.legend()
plt.show()

####### Block 6
def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()

# plot image and keypoints
image_index = 360
image = dataset_testing.returnImage(image_index)
keypoints = dataset_testing.returnKeypoints(image_index)

from torchvision.utils import draw_keypoints
res = draw_keypoints(image, keypoints[:,0:2].unsqueeze(0), colors="blue", radius=10)
show(res)

# predict the keypoints
with torch.no_grad():
    X_train, y_train = dataset_testing.__getitem__(image_index - (sequence_length+1))
    train_size = int(sequence_length * split_ratio)
    X_train = X_train.unsqueeze(0)
    X_train = X_train.view(-1, X_train.size(2), X_train.size(3))
    #train_plot = np.ones_like(sequence_length) * np.nan
    y_pred = LSTM_model(X_train)
    keypoints_predicted = y_pred[:, -1, :].squeeze()
    #test_plot[train_size+4:len(timeseries)] = model(X_test)[:, -1, :].squeeze()

# make sure you pick the last keypoint in the sequence
res_2 = draw_keypoints(image, keypoints_predicted.unsqueeze(0), colors="red", radius=10)

show(res_2)

#plt.plot(sequence_length, label='function to optimize')
#plt.plot(train_plot, c='r', label='train prediction')
#plt.plot(test_plot, c='g', label='test prediction')
#plt.legend()
#plt.show()

'''
---------Task 5---------
##  What would happen if several person would be in the frame?
If there were multiple people in the frame, the model will be unable to differentiate between the two and either will only develop keypoints for one person,
or if the people were close together it may think the people are a singular person and subsequently make keypoints between the two of them.

##  How would you change the algorithm to account for multiple people?
To account for multiple people, a differential between the two would be required, with a sort of person detection system that is accurate enough to separate 
the two people if they are close together. This would need to be implemented in the 'preprocessData' function or just prior to it, before identifying the 
keypoints of the model.

##  If the LSTM has a poor performance on the testing set, how would you explain it?
There are a few different issues that could result in poor test set performance, including overfitting, if the model was created to be too simple, or from
minimal training data (ie. if there are only 200 photos rather than ~1200).

##  At what epochs shall we stop the training of the LSTM?
Training should stop when the test loss converges to a singular value, and tuned so that it is not continuing for too long after it starts to converge, which
may cause overfitting to occur.

##  List all the parts of the code that needs to be changed to predict all the keypoints at once using the LSTM (and what would you change)
To predict all the keypoints, you would have to:
    - Make a loop of 'getItem' so that it appends all keypoints to one variable, or modify it so it extracts all the keypoints in one run.
    - Adjust the sizing of the LSTM class so that it suits the amount of keypoints you have.
    - Adjust the plot so that all of the keypoints are included.
'''