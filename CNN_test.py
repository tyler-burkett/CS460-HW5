import dataset
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from CNN import CNN

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    neural_net = CNN()
    neural_net.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(neural_net.parameters(), lr=0.1)

    # Import Corgi images
    train_set = dataset.read_train_sets("./data/training_data", 500, ['pembroke', 'cardigan'], 0)
    test_set = dataset.read_train_sets("./data/testing_data", 500, ['pembroke', 'cardigan'], 0)

    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(zip(train_set.train.images(), train_set.train.labels()), 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = torch.from_numpy(np.expand_dims(np.swapaxes(np.swapaxes(data[0], 2, 1), 1, 0), axis=0)).to(device)
            labels = torch.from_numpy(np.where(data[1] == 1)[0]).to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = neural_net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

    print('Finished Training')
    path = "./net_params.pth"
    torch.save(neural_net.state_dict(), path)

    total = len(test_set.train._images)
    correct = 0
    for i, data in enumerate(zip(test_set.train.images(), test_set.train.labels()), 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs = torch.from_numpy(np.expand_dims(np.swapaxes(np.swapaxes(data[0], 2, 1), 1, 0), axis=0)).to(device)
        labels = np.where(data[1] == 1)[0][0]

        # forward
        outputs = neural_net(inputs)
        if torch.max(outputs, 0)[1][0].item() == labels:
            correct += 1

    print("Accuracy: {}".format(correct/total))
