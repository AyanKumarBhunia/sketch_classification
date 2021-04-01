
from Networks import *
from torch import optim
import torch
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torchvision.utils import save_image


class Sketch_Classification(nn.Module):
    def __init__(self, hp):
        super(Sketch_Classification, self).__init__()
        self.Network = eval(hp.backbone_name + '_Network(hp)')
        self.train_params = self.parameters()
        self.optimizer = optim.Adam(self.train_params, hp.learning_rate)
        self.loss = nn.CrossEntropyLoss()
        self.hp = hp

    def train_model(self, batch):
        self.train()
        self.optimizer.zero_grad()
        img = batch['sketch_img'].to(device)
        output = self.forward(img)
        loss = self.loss(output, batch['sketch_label'].to(device))
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def forward(self, x):
        return self.Network(x)

    def evaluate(self, dataloader_Test):
        self.eval()
        correct = 0
        test_loss = 0
        start_time = time.time()
        for i_batch, batch in enumerate(dataloader_Test):
            img = batch['sketch_img'].to(device)
            output = self.forward(img)
            test_loss += self.loss(output, batch['sketch_label'].to(device)).item()
            prediction = output.argmax(dim=1, keepdim=True).to('cpu')
            correct += prediction.eq(batch['sketch_label'].view_as(prediction)).sum().item()

        test_loss /= len(dataloader_Test.dataset)
        accuracy = 100. * correct / len(dataloader_Test.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Time_Takes: {}\n'.format(
            test_loss, correct, len(dataloader_Test.dataset), accuracy, (time.time() - start_time) ))

        return accuracy


