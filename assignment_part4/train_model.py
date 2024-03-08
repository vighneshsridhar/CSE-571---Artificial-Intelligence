from Data_Loaders import Data_Loaders
from Networks import Action_Conditioned_FF

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def train_model(no_epochs):

    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    model = Action_Conditioned_FF()


    losses = []
    # loss_function = nn.BCELoss()
    loss_function = nn.MSELoss()
    #min_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
    #losses.append(min_loss)
    # optimizer = optim.SGD(model.parameters(), lr=0.01)
    optimizer = optim.Adam(model.parameters(), lr=0.01)


    for epoch_i in range(no_epochs):
        running_loss = 0
        correct = 0
        total = 0
        model.train()
        for idx, sample in enumerate(data_loaders.train_loader): # sample['input'] and sample['label']

            input, label = sample['input'], sample['label']
            # input = torch.tensor(sample['input'], dtype=torch.float32)
            # label = torch.tensor(sample['label'], dtype=torch.float32)

            optimizer.zero_grad()

            output = model(input)
            if output.ndimension() == 2 and output.size(1) == 1:
                output = output.squeeze(1)
            loss = loss_function(output, label)
            # loss = loss_function(output, label)
            # print ("output, label = ", output, label)
            for o, l in zip(output, label):
                pred = 0 if o.item() < 0.25 else 1
                if pred == l.item():
                    correct += 1
                total += 1

            loss.backward()
            optimizer.step()

            losses.append(loss)
            running_loss += loss.item()

            #if idx % 2000 == 1999:    # print every 2000 mini-batches
                #print(f'[{epoch + 1}, {idx + 1:5d}] loss: {running_loss / 2000:.3f}')
                # running_loss = 0.0


        # print(f'[{epoch_i + 1}, {idx + 1:5d}] loss: {running_loss / 2000:.3f}')
        accuracy = (correct/total)*100
        print ("epoch_i = ", epoch_i, "running loss = ", running_loss, "accuracy = ", accuracy)

    print('Finished Training')

    print("test loss = ", model.evaluate(model, data_loaders.test_loader, loss_function))
    torch.save(model.state_dict(), 'saved/saved_model.pkl')



if __name__ == '__main__':
    no_epochs = 100
    train_model(no_epochs)
