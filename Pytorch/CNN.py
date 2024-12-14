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
print(training_data[0])

plt.imshow(training_data[2][0], cmap="gray") #make sure you access the matrix(image) alone and not the one hot included
plt.show()





class Model(nn.module):

    def __init__(self):
        super.__init__()
        self.conv1 = nn.Conv2d(1,32,5) #1-in channels (no rgb hence 1), 32-feature maps, 5-kernel size
        self.conv2 = nn.Conv2d(1,32,5)
        self.conv3 = nn.Conv2d(1,32,5)


    
    