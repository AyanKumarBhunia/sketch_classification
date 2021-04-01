import torch
from models import *
from dataset import get_dataloader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import argparse



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Skecth_Classification')
    parser.add_argument('--backbone_name', type=str, default='Resnet', help='VGG / InceptionV3/ Resnet')
    parser.add_argument('--pool_method', type=str, default='AdaptiveAvgPool2d', help='AdaptiveMaxPool2d / AdaptiveAvgPool2d / AvgPool2d')
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--nThreads', type=int, default=8)
    parser.add_argument('--splitTrain', type=float, default=0.8)

    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--eval_freq_iter', type=int, default=200)
    parser.add_argument('--print_freq_iter', type=int, default=5)

    hp = parser.parse_args()
    dataloader_Train, dataloader_Test = get_dataloader(hp)
    print(hp)

    #######################################################################
    ############################## End Load Data ##########################
    model = Sketch_Classification(hp)
    model.to(device)
    model.load_state_dict(torch.load('model_best_TUBerlin.pth', map_location=device))
    with torch.no_grad():
        accuracy = model.evaluate(dataloader_Test)

    step = 0
    best_accuracy = 0

    for epoch in range(hp.max_epoch):

        for i_batch, batch in enumerate(dataloader_Train):
            loss = model.train_model(batch)
            step += 1

            if (step + 1) % hp.print_freq_iter == 0:
                print('Epoch: {}, Iter: {}, Steps: {}, Loss: {}, Best Accuracy: {}'.format(epoch, i_batch, step, loss, best_accuracy))

            if (step + 1) % hp.eval_freq_iter == 0:
                with torch.no_grad():
                    accuracy = model.evaluate(dataloader_Test)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    torch.save(model.state_dict(), 'model_best_TUBerlin.pth')
