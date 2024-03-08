import torch
import torch.nn as nn
from Data_Loaders import Data_Loaders
import torch.optim as optim

class Action_Conditioned_FF(nn.Module):
    def __init__(self):
# STUDENTS: __init__() must initiatize nn.Module and define your network's
# custom architecture
        super(Action_Conditioned_FF, self).__init__()
        self.input_to_hidden = nn.Linear(6, 3)
        self.nonlinear_activation = nn.Sigmoid()
        self.hidden_to_output = nn.Linear(3, 1)

        pass

    def forward(self, input):
# STUDENTS: forward() must complete a single forward pass through your network
# and return the output which should be a tensor
        hidden = self.input_to_hidden(input)
        hidden = self.nonlinear_activation(hidden)
        output = self.hidden_to_output(hidden)
        return output


    def evaluate(self, model, test_loader, loss_function):
# STUDENTS: evaluate() must return the loss (a value, not a tensor) over your testing dataset. Keep in
# mind that we do not need to keep track of any gradients while evaluating the
# model. loss_function will be a PyTorch loss function which takes as argument the model's
# output and the desired output.

        loss = 0
        model = model.eval()

        for idx, sample in enumerate(test_loader):
            output = model(sample['input'])
            print ("sample input shape = ", sample['input'].shape)
            target = sample['label']
            loss += loss_function(output, target)

        loss = loss/len(test_loader)

        return loss

def main():
    model = Action_Conditioned_FF()

model = Action_Conditioned_FF()
optimizer = optim.SGD(model.parameters(), lr=0.01)
num_epochs = 10
batch_size = 16
data_loaders = Data_Loaders(batch_size)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for idx, sample in enumerate(data_loaders.train_loader):
        optimizer.zero_grad()
        output = model(sample['input'])
        target = sample['label']
        target = target.view(-1, 1)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loaders.train_loader)
    print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")

model = Action_Conditioned_FF()
loss_function = nn.MSELoss()
test_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
print(f'Test Loss: {test_loss.item():.4f}')




if __name__ == '__main__':
    main()
