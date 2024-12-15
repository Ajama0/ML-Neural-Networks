import os
import cv2
from tqdm import tqdm
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

REBUILD_DATA = False

class DogsVsCats():
    IMG_SIZE = 50   
    CATS = "PetImages\Cat"
    DOGS = "PetImages\Dog"
    

    LABELS = {CATS: 0, DOGS: 1}
   

    Training_data = []
    catsCount = 0
    dogsCount=0

    def make_training_data(self):
            #iterating through each image, in each folder and applying gray scale
        for labels in self.LABELS: #each label represents a path to cats and dogs folder
            print(labels)
            for f in tqdm(os.listdir(labels)):#iterating through each image in directory of cats/dogs
                    try:
                        path = os.path.join(labels,f) #exact path to the image
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) #read the image specifically for that label and convert it to grayscale
                        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))#resize the image by passing in the veritcal and horizontal values in our case 50x50
                        self.Training_data.append([np.array(img), np.eye(2)[self.LABELS[labels]]])
                        #we appemnd the images as a 2d array, and the label is a one hot encoded,
                            
                        """we appemnd the images as a 2d array, and the label is a one hot encoded,
                            np.eye() takes a parameter to define the number of classes i.e np.eye(2)= 2 classes
                            np.eye(2)[] means we can define which value is hot based on the sample
                            np.eye(2)[0(label of cat and dog)] means we have a [1,0]
                            np.eye(2)[1] means we produce a [0,1]

                            """
                            #labels represent path to cats so for every cat 
                        if labels == self.CATS:
                            self.catsCount +=1

                        elif labels == self.DOGS:
                            self.dogsCount +=1

                    except Exception as e:
                        pass    


        #if our try executes and we return the training data

        #shuffle the data to prevent the model from overfitting

        np.random.shuffle(self.Training_data)
        with open("training_data.pkl", "wb") as f:
            pickle.dump(self.Training_data, f)
        print("catCount: ", self.catsCount)
        print("dogCount: ", self.dogsCount)
                    
        
if REBUILD_DATA:
    dogsvscats = DogsVsCats()
    dogsvscats.make_training_data()

with open("training_data.pkl", "rb") as f:
    training_data = pickle.load(f)
print(len(training_data))
#print(training_data[0])

#plt.imshow(training_data[2][0], cmap="gray") #make sure you access the matrix(image) alone and not the one hot included
#plt.show()





class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,32,5) #1-in channels (no rgb hence 1), 32-feature maps, 5-kernel size
        self.conv2 = nn.Conv2d(32,64,5)
        self.conv3 = nn.Conv2d(64,128,5) #we must flatten the 128 feature maps for us to use it in a FCNN last layer
        #pooling layer
        self.pool1 = nn.MaxPool2d((2,2))
        self.pool2 = nn.MaxPool2d((2,2))
        self.pool3 = nn.MaxPool2d((2,2))

        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512,2)


    

    


    def forward(self, x):
        #the conv1 returns an object with state defined as the parameters of filters and kernal size
        #automatically pytorch calls forward method of nn.module whenever a value is passed to it
        #its like you writing (instance of Model)output = self.conv1.forward(x) 
        #conv1 is is an instance of conv2d and conv1(x) calls the forward method of conv2d class
        x = F.relu(self.conv1(x)) #returns a feature map for each filter
        x = self.pool1(x) #apply a pooling layer of 2x2 on each feature map to reduce dimensions
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        #print(x.shape)
        #the last layer before passig it to a fcnn
        #it contains many feature sets which need to be flattend to match the linear layer
        #flatten the feature sets into a 1d vector
        x = x.flatten(start_dim = 1) #flatten the shape, as we pass in a dummy value, and see what the shape is at that current point for the dimensions that are the same as the intitial input
        #print(x.shape)#print the shape before passing it to the FCNN. so that when we call fc1(x) we know what input it receives so that it can create the corresponding weight matrix
        x = F.relu(self.fc1(x))
        x = self.fc2(x) #last layer will only consist of a softmax activation and not an activation value

        return F.softmax(x,dim=1) # apply softmax to the batch, but dim = 1 refers to each sample vector at a time
    
#global object
net = Net()

#x = torch.rand(1,1,50,50) #the dummy returned to us a shape of [1,512] so we know the input neurons after the conv3d is a 1d vector with 512 neurons
#print(net(x))




optimizer = optim.Adam(net.parameters(), lr = 0.001)
loss_function = nn.MSELoss()


#iterate through the training data, and return the image matrix, [0] referes to the the first element in the lol containing sample matrix and label,
#reshape it into the dimesnions and apply torch.tensor to perform operations on them
X = torch.tensor(np.array([i[0] for i in training_data]),dtype=torch.float32).view(-1,50,50)
print(X.shape)
X = X/255.0 #divide pixels by 255 to be range [0,1]. we normalize pixels to reduce computations when doing convolutions
y = torch.tensor(np.array([i[1] for i in training_data]),dtype=torch.float32) #create a new list of labels for every sample

#define train and test test
valid_pct = 0.1 #10% of the data will be used for test
val_size = int(len(X)*valid_pct) #will return 2494

train_x =X[:-val_size]#iterate through training data and stop before the last 2494 samples
print("new training data after dividing from test set", len(train_x))
train_y = y[:-val_size]


test_x = X[-val_size:] #iterate the last 2494 images
test_y = y[-val_size:]

BATCH_SIZE = 100
EPOCHS = 1

for epochs in range(EPOCHS):
    #iterate through the train data in a step size of the batches
    for i in tqdm(range(0, len(train_x), BATCH_SIZE)):
        #iterate until end of trainset, collecting a batch of 100 samples. each iter is a 100 samples
        batch_x = train_x[i:i+BATCH_SIZE].view(-1,1,50,50)
        batch_y = train_y[i:i+BATCH_SIZE]


        net.zero_grad() #for each batch the gradient is reset

        output = net(batch_x)
        loss = loss_function(output, batch_y)
        loss.backward()
        optimizer.step()

print("here")

Correct = 0
Total = 0


with torch.no_grad:
    






    
    