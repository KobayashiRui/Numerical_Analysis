# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

accuracies_train = []
accuracies_test  = []
epoch_list = []

with open('./epoch_list.txt',"r") as f:
    line = f.readline().strip()
    epoch_list.append(line)
    while line:
        epoch_list.append(line)
        line = f.readline().strip()

epoch_list = map(float,epoch_list)
print(epoch_list)

with open('./accuracies_train_data.txt',"r") as f:
    line = f.readline().strip()
    accuracies_train.append(line)
    while line:
        accuracies_train.append(line)
        line = f.readline().strip()

accuracies_train = map(float,accuracies_train)
print(accuracies_train)

with open('./accuracies_data.txt',"r") as f:
    line = f.readline().strip()
    accuracies_test.append(line)
    while line:
        accuracies_test.append(line)
        line = f.readline().strip()

accuracies_test = map(float,accuracies_test)
print(accuracies_test)

plt.title('Accuracy per epoch')
plt.xlabel("epoch")
plt.ylabel("Accuracy [%]")
plt.xlim([-5,205])
plt.ylim([70,100])
plt.plot(epoch_list,accuracies_train,marker='o',label="train_accuracy")
plt.plot(epoch_list,accuracies_test,marker='o',label="test_accuracy")
plt.legend(loc="lower right")

plt.show()
