import torch
import torch.nn as nn


class Action_Conditioned_FF(nn.Module):
    def __init__(self):
# STUDENTS: __init__() must initiatize nn.Module and define your network's
# custom architecture
        super(Action_Conditioned_FF, self).__init__()
        self.input_to_hidden1 = nn.Linear(6, 150)
        self.output_activation = nn.Sigmoid()
        self.nonlinear_activation = nn.ReLU()
        #self.hidden1_to_hidden2 = nn.Linear(150, 8)
        self.hidden2_to_output = nn.Linear(150, 1)


    def forward(self, input):
# STUDENTS: forward() must complete a single forward pass through your network
# and return the output which should be a tensor
        hidden = self.input_to_hidden1(input)
        hidden = self.nonlinear_activation(hidden)
        # hidden2 = self.hidden1_to_hidden2(hidden)
        # hidden2 = self.nonlinear_activation(hidden2)
        output = self.hidden2_to_output(hidden)
        # output = self.output_activation(output)
        return output


    def evaluate(self, model, test_loader, loss_function):
# STUDENTS: evaluate() must return the loss (a value, not a tensor) over your testing dataset. Keep in
# mind that we do not need to keep track of any gradients while evaluating the
# model. loss_function will be a PyTorch loss function which takes as argument the model's
# output and the desired output.

        loss = 0
        correct = 0
        total = 0

        model.eval()
        for idx, sample in enumerate(test_loader):
            output = model(sample['input'])
            # print ("sample input shape = ", sample['input'].shape)
            target = sample['label']
            # print ("output = ", output, " target = ", target)

            if output.ndimension() == 2 and output.size(1) == 1:
                output = output.squeeze(1)
            loss += loss_function(output, target)

            for o, l in zip(output, target):
                pred = 0 if o.item() < 0.25 else 1
                if pred == l.item():
                    correct += 1
                total += 1

        loss = loss / len(test_loader)
        accuracy = (correct / total) * 100
        # print("testing loss = ", loss, "accuracy = ", accuracy)
        return loss


def main():
    model = Action_Conditioned_FF()
    # data_loaders = Data_Loaders(batch_size=16)
    # print(model.evaluate(model, data_loaders.test_loader, nn.MSELoss()))


if __name__ == '__main__':
    main()
