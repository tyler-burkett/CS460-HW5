import dataset
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from CNN import CNN
import matplotlib.pyplot as plt

def imgshow(img, swap=True):
    if swap:
        img = np.transpose(img, (1, 2, 0))
    fig, axes = plt.subplots(1, 4)
    axes[0].imshow(img[:,:,0], cmap=plt.cm.Reds_r)
    axes[1].imshow(img[:,:,1], cmap=plt.cm.Greens_r)
    axes[2].imshow(img[:,:,2], cmap=plt.cm.Blues_r)
    axes[3].imshow(img)
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    neural_net = CNN()
    neural_net.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(neural_net.parameters())

    # Import Corgi images
    train_set = dataset.read_train_sets("./data/training_data", 100, ['pembroke', 'cardigan'], 0)
    test_set = dataset.read_train_sets("./data/testing_data", 100, ['pembroke', 'cardigan'], 0)

    train_set.train._images = train_set.train._images[:20]
    train_set.train._labels = train_set.train._labels[:20]
    neural_net.train()
    for epoch in range(20):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(zip(train_set.train.images(), train_set.train.labels()), 0):
            # get the inputs and label
            #imgshow(np.transpose(data[0], (2, 0, 1)))
            inputs = torch.from_numpy(np.expand_dims(np.transpose(data[0], (2, 0, 1)), axis=0)).to(device)
            label = torch.from_numpy(np.array([data[1].argmax()])).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            output = neural_net(inputs)
            loss = loss_function(output, label)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i == len(train_set.train._images)-1:    # print every 2000 mini-batches
                print('[%d] loss: %.3f' %
                      (epoch + 1, running_loss / len(train_set.train._images)))
        if running_loss / len(train_set.train._images) < 0.001:
            break

    corgi_names = ['pembroke', 'cardigan']

    total = len(test_set.train._images)
    correct = 0
    neural_net.eval()
    for i, data in enumerate(zip(test_set.train.images(), test_set.train.labels()), 0):
        # get the inputs and label
        inputs = torch.from_numpy(np.expand_dims(np.transpose(data[0], (2, 0, 1)), axis=0)).to(device)
        label = data[1].argmax()

        # forward
        output = neural_net(inputs)
        output = output.cpu().detach()
        output_class = output.numpy().argmax()
        print(output.numpy())
        print("Actual {}, Predicted: {}".format(corgi_names[label], corgi_names[output_class]))
        if output_class == label:
            correct += 1

    print("Accuracy: {}".format(correct/total))
