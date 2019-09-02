import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional
import torchvision.datasets
import torchvision.transforms
from torch.autograd import Variable

from torch import np   # this is torch's wrapper for numpy
from matplotlib import pyplot
from matplotlib.pyplot import subplot
from models.lenet5 import LeNet5
from models.fullyconnected import FCNet
from sklearn.metrics import accuracy_score


### TILL HERE I WOULD GIVE


#configs
batch_size = 32
num_epochs = 10




"""
CONFLICT ZONE 1
##CNN
transformImg = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                                               
##P1 COMMITS PUSHES AND MERGES

##FCNET
transformImg = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

##P2 COMMITS ---> PULL FROM MASTER --> RECEIVES CONFLICT
"""



"""
LOCAL RESOLVE CONFLICT - P2

VARIABLE TO RESOLVE CONFLICT 
ARG PARSER
mode = "cnn"
or
mode = "fc"
mode = arg.parser(arg = "mode")


transformImg = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) if mode = "cnn" else torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

## P2 HAS TO COMMIT (THEREFORE RESOLVES CONFLICT) ---> PUSH TO P2 BRANCH --> MERGE TO MASTER(PR)
"""



### P1 PULLS MASTER INTO HIS BRANCH
### P1 START REGION
train_dataset = torchvision.datasets.MNIST(root='../../data',
                                           train=True,
                                           transform=transformImg,
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data',
                                          train=False,
                                          transform=transformImg)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

### P1 START REGION ---> PUSH TO HIS BRANCH --> MERGE TO MASTER (PR)



#OPTIONAL CODE
# print("Training data dimensions: ", train_dataset.train_data.shape)
# print("Test data dimensions: ", test_dataset.test_data.shape)
#
# print("\nAn image in matrix format looks as follows: ", train_dataset.train_data[0])
#
#
# fig1 = train_dataset.train_data[0].numpy()
# fig2 = train_dataset.train_data[2500].numpy()
# fig3 = train_dataset.train_data[25000].numpy()
# fig4 = train_dataset.train_data[59999].numpy()
# subplot(2,2,1), pyplot.imshow(fig1)
# subplot(2,2,2), pyplot.imshow(fig2)
# subplot(2,2,3), pyplot.imshow(fig3)
# subplot(2,2,4), pyplot.imshow(fig4)




"""
CONFLICT ZONE 2

##P2 PULL FROM MASTER INTO HIS BRANCH --> COMMITS LINE 1 ---> PUSH TO HIS BRANCH ---> MERGES WITH MASTER 
LINE1 : model = LeNet5()

##P1 DOES NOT PULL --> COMMITES LINE 2 ---> PUSH TO HIS BRANCH ---> PR CONFLICT
LINE 2: model = FCNet()
"""

"""
REMOTE RESOLVE CONFLICT
model = LeNet5() if mode = "cnn" else FCNet()
"""

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

training_accuracy = []

"""
UTIL FUNCTION TO RESOLVE CONFLICT
input_reshape = lambda x,mode : x.reshape(-1, 28,28) if mode == "cnn" else x.reshape(-1,28*28)
"""


for epoch in range(num_epochs):
    print("Epoch:", epoch)
    for batch_num, train_batch in enumerate(train_loader):

        images, labels = train_batch

        """
        CONFLICT ZONE 3
        P1 : BRANCH 1 inputs = Variable(images.reshape(-1, 28*28))
        P2 : BRANCH 2 inputs = Variable(images.reshape(-1, 28,28))
        """

        """
        RESOLVE CONFLICT
        inputs = Variable(input_reshape(mode))
        """
        targets = Variable(labels)

        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        print("Batch Num: {} Loss: {}".format(batch_num, loss))

    ##training accuracy after each epoch
        accuracy = 0.0
        num_batches = 0
        for batch_num, training_batch in enumerate(train_loader):        # 'enumerate' is a super helpful function
            num_batches += 1
            images, labels = training_batch

            """
            CONFLICT ZONE 3
            P1 : BRANCH 1 inputs = Variable(images.reshape(-1, 28*28))
            P2 : BRANCH 2 inputs = Variable(images.reshape(-1, 28,28))
            """

            """
            RESOLVE CONFLICT
            inputs = Variable(input_reshape(mode))
            """
            targets = labels.numpy()
            inputs = Variable(inputs)
            outputs = model(inputs)
            outputs = outputs.data.numpy()
            predictions = np.argmax(outputs, axis = 1)
            accuracy += accuracy_score(targets, predictions)
            training_accuracy.append(accuracy/num_batches)


## test on testing dataset

with torch.no_grad():
    correct = 0
    total = 0
    for test_batch in test_loader:
        images, labels = test_batch

        """
        CONFLICT ZONE 3
        P1 : BRANCH 1 inputs = Variable(images.reshape(-1, 28*28))
        P2 : BRANCH 2 inputs = Variable(images.reshape(-1, 28,28))
        """

        """
        RESOLVE CONFLICT
        inputs = Variable(input_reshape(mode))
        """
        images = images.reshape(-1, 28*28)
        labels = labels
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))




##OPTIONAL display epochs and accuracy using Matplotlib











