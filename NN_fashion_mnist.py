#-*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt



class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        #画像の畳込み
        self.head = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3),stride=1, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=1),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3),stride=1,padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3),stride=1,padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=1),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3),stride=1,padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
                )
        
        #畳み込み結果から全結合
        self.tail = nn.Sequential(
                nn.Linear(1024, 516),
                nn.ReLU(),
                nn.Linear(516, 320),
                nn.ReLU(),
                nn.Linear(320, 228),
                nn.ReLU(),
                nn.Linear(228,120),
                nn.ReLU(),
                nn.Linear(120,80),
                nn.ReLU(),
                nn.Linear(80,20),
                nn.ReLU(),
                nn.Linear(20,10)
                )

    def __call__(self, x):
        h = self.head(x)
        h = h.view(-1, 1024)
        h = F.dropout(h,0.5,training=self.training)
        h = self.tail(h)
        h = F.dropout(h,0.5,training=self.training)
        y = F.log_softmax(h,dim=1)#dimensionを指定する
        return y

#学習


#教師データ
fashion_mnist_data = datasets.FashionMNIST('./data/fashion-mnist', train=True,download=True,transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=fashion_mnist_data,batch_size=50,shuffle=True)

fashion_mnist_data_test = datasets.FashionMNIST('./data/fashion-mnist',transform=transforms.ToTensor(), train=False, download=True)

test_loader = torch.utils.data.DataLoader(dataset=fashion_mnist_data_test,batch_size=50,shuffle=True)


model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
using_cuda = torch.cuda.is_available()
if using_cuda:
    print("using cuda")
    model.cuda()
    criterion.cuda()

#学習開始
EPOCH_NUM = 200
#model define
accuracies= []
accuracies_train = []
loss_list = []
epoch_list = []
print("Train Start")
for epoch in range(EPOCH_NUM+1):
    model.train()
    #mini bach
    total_loss = 0
    print("####### epoch : {} ######".format(epoch))
    for i, data in enumerate(train_loader):
        x, t = data
        x, t = Variable(x.cuda()),Variable(t.cuda())
        optimizer.zero_grad()
        y = model(x)
        loss = criterion(y, t)
        #total_loss += loss.data[0]
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()

    if (epoch)%10 ==0:
        model.eval()
        epoch_list.append(epoch)
        loss_list.append(total_loss)
        print("epoch:\t{}\ttotal loss:\t{}".format(epoch, total_loss))

        n_true = 0
        for batch, labels in test_loader:
            if using_cuda:
                output = model(Variable(batch.cuda()))
            else:
                output = model(Variable(batch))
            _, predicted = torch.max(output.data, 1)
            if using_cuda:
                y_predicted = predicted.cpu().numpy()
            else:
                y_predicted = predicted.numpy()
            n_true += np.sum(y_predicted == labels.numpy())
    
        total = len(fashion_mnist_data_test)
        accuracy = 100.0 * n_true / total
        print('epoch: {0}, accuracy: {1}'.format(epoch, (accuracy)))
        accuracies.append(accuracy)
        
        n_true_t = 0
        for batch_t, labels_t in train_loader:
            if using_cuda:
                output_t = model(Variable(batch_t.cuda()))
            else:
                output_t = model(Variable(batch_t))
            _, predicted_t = torch.max(output_t.data, 1)
            if using_cuda:
                y_predicted_t = predicted_t.cpu().numpy()
            else:
                y_predicted_t = predicted_t.numpy()
            n_true_t += np.sum(y_predicted_t == labels_t.numpy())
    
        total_t = len(fashion_mnist_data)
        accuracy_t = 100.0 * n_true_t / total_t
        print('train epoch: {0}, accuracy: {1}'.format(epoch, (accuracy_t)))
        accuracies_train.append(accuracy_t)

print("Finish")

with open('./loss_data.txt', mode='w') as f:
    for data in loss_list:
        f.write("{}\n".format(data))

with open('./accuracies_data.txt', mode='w') as f:
    for data_acc in accuracies:
        f.write("{}\n".format(data_acc))

with open('./accuracies_train_data.txt', mode='w') as f:
    for data_acct in accuracies_train:
        f.write("{}\n".format(data_acct))

with open('./epoch_list.txt', mode='w') as f:
    for epoch_data in epoch_list:
        f.write("{}\n".format(epoch_data))


#plt.plot(epoch_list, loss_list)
plt.plot(epoch_list, accuracies)
plt.plot(epoch_list, accuracies_train)
plt.show()
