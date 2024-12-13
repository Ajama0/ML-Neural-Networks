import torch
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#seperate train and test data

train = datasets.MNIST(root = "", train=True, download=True,
                       transform= transforms.Compose([transforms.ToTensor()]))


test = datasets.MNIST(root= "", train=False, download=True,
                       transform= transforms.Compose([transforms.ToTensor()]))

#load in the dataset and assign it to a variable
train_data = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
test_data = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10) #10 output neurons 0-9


    def forward(self,x):
        x = F.relu(self.fc1(x))   
        x = F.relu(self.fc2(x)) 
        x = F.relu(self.fc3(x))  
        x = self.fc4(x) 

        return F.log_softmax(x, dim=1)  
    

net = Net()
#x= torch.randn((28,28))
#output = net(x.view(-1,28*28))
#print(output)


optimizer = optim.Adam(net.parameters(), lr = 0.001)    

EPOCHS = 3 

for epochs in range(EPOCHS):
    #Loop through data in train data
    for data in train_data:
    #data represents a batch of samples(images) and labels for each sample
     #x, y = the feature sets, and labels. so we unpack that here
        X, y = data
        net.zero_grad() # reo
        #first pass in the data

        #net() allows us to automatically call forward, when inherited from nn.module, forward is the default callable behaviour
        #now when we pass in net(x.view(1,-1)) we reshape the image to fit the input size (784), then call the forward pass
        output = net(X.view(-1,784))

        #calculate the loss

        #what was the loss of our prediction
        loss = F.nll_loss(output, y)

        loss.backward() #this traverses through the computational graph and calculates the partial derivate of each weight wrt g.d

        optimizer.step() #this adjusts our weight for us based on the weight wrt to g.d
        

    print(loss)


#we want to evaluate the models performance, and dont require any gradient computations here.
correct = 0
total=0
with torch.no_grad():
    for data in train_data:
        X,y = data
        otuput = net(X.view(-1,784)) #this calculates the softmax probabilites for the outputs
        #so for each sample softmax values, the argmax would be our predicted class
        for idx, i in enumerate(output):
            #for each index of a row(softmax values) and the row vector itself
            #if the index of the predicted class, is equal to the index of its corresponding y true value
            if torch.argmax(i)==y[idx]:
                correct +=1
            total+=1  

accuracy = round(correct / total,3)
print(f"Accuracy: {accuracy}")              








       

        
       


    
    

    
