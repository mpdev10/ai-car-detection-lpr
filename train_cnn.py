import argparse
import pickle as pkl

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from cnn.cnn import CNN

parser = argparse.ArgumentParser(
    description='Convolutional Neural Network Training With Pytorch')

parser.add_argument('--num_epochs', default=25, type=int, help='The number of epochs')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
parser.add_argument('--learning_rate', default=0.001, type=float, help='Learning rate for training')
parser.add_argument('--dataset', type=str, default='train.pkl', help='Path of dataset')
parser.add_argument('--store_path', type=str, default='models/', help='Path to save trained model')
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--model', default='models/cnn.ckpt', type=str)

args = parser.parse_args()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")

if args.use_cuda and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    print("Use Cuda.")

if __name__ == '__main__':

    data, labels = pkl.load(open(args.dataset, 'rb'))

    transform = transforms.Compose([transforms.ToTensor()])

    train_data = torch.stack([torch.Tensor(i) for i in data])
    train_labels = torch.from_numpy(np.reshape(labels, (-1))).long()

    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False)
    model = CNN()

    if args.model is not None:
        state_dict = torch.load(args.model, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)
        print("Loaded model: " + args.model)

    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

    total_step = len(train_loader)
    loss_list = []

    acc_list = []

    for epoch in range(args.num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, args.num_epochs, i + 1, total_step, loss.item(),
                              (correct / total) * 100))

    model.eval()
    torch.save(model.state_dict(), args.store_path + 'cnn.ckpt')
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the model on the test images: {} %'.format((correct / total) * 100))
