import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

# sklearn libraries
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import os
import argparse
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from read_data import read_data
from visualize import view_samples, plot_confusion_matrix
from dataset_loader import GTSRB
from models import initialize_model
from train import train, validate

device = torch.device("cpu")


# '''
parser = argparse.ArgumentParser(description='Traffic Sign Detection Training')
parser.add_argument('--use_cuda', type=bool, default=False, help='device to train on')
parser.add_argument('--samples', type=bool, default=False, help='See sample images')
parser.add_argument('--view_data_counts', type=bool, default=False, help='Visualize data distribution')
parser.add_argument('--model', type=str, default='densenet', help='resnet,vgg,densenet,inception')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train on')
parser.add_argument('--train', default=True, type=bool, help='train the model')

opt = parser.parse_args()


if opt.use_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# '''

root = os.path.join('../','input','gtsrb-german-traffic-sign')
Cells, labels = read_data(root)


if opt.samples:
    view_samples(Cells, labels)


if opt.train:
    X_train, X_val, y_train, y_val = train_test_split(Cells, labels, test_size=0.2, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

    model_ft, in_size = initialize_model(opt.model, num_classes=43, feature_extract=True)
    model_ft.to(device)

    # we use Adam optimizer, use cross entropy loss as our loss function
    optimizer = optim.Adam(model_ft.parameters(), lr=1e-3, betas=(0.9,0.999))
    criterion = nn.CrossEntropyLoss().to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                            mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, 
                                                            threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)


    train_transform = transforms.Compose([
                                        transforms.Resize(in_size),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        ])
    
    # define the transformation of the val images.
    val_transform = transforms.Compose([
                                        transforms.Resize(in_size),
                                        transforms.ToTensor(),
                                        ])

    # Define the training set using the table train_df and using our defined transitions (train_transform)
    training_set = GTSRB(X_train, y_train, transform=train_transform)
    train_loader = DataLoader(training_set, batch_size=16, shuffle=True, num_workers=4)
    
    # Same for the validation set:
    validation_set = GTSRB(X_val, y_val, transform=val_transform)
    val_loader = DataLoader(validation_set, batch_size=16, shuffle=False, num_workers=4)

    # Test Set
    test_set = GTSRB(X_test, y_test, transform=val_transform)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=4)

    import time
    since = time.time()
    epoch_num = 20
    best_val_acc = 0
    total_loss_val, total_acc_val = [],[]
    for epoch in tqdm((range(1, epoch_num+1))):
        print('\n'*2)
        loss_train, acc_train, total_loss_train, total_acc_train = train(train_loader, model_ft, criterion, optimizer, epoch, device)
        loss_val, acc_val = validate(val_loader, model_ft, criterion, optimizer, epoch, device)
        total_loss_val.append(loss_val)
        total_acc_val.append(acc_val)
        scheduler.step(loss_val)
        if acc_val > best_val_acc:
            best_val_acc = acc_val
            print('\n')
            print('*****************************************************')
            print('best record: [epoch %d], [val loss %.5f], [val acc %.5f]' % (epoch, loss_val, acc_val))
            print('*****************************************************')

    print ('Time Taken: ',time.time()-since)

    fig = plt.figure(num = 2)
    fig1 = fig.add_subplot(2,1,1)
    fig2 = fig.add_subplot(2,1,2)
    fig1.plot(total_loss_train, label = 'training loss')
    fig1.plot(total_acc_train, label = 'training accuracy')
    fig2.plot(total_loss_val, label = 'validation loss')
    fig2.plot(total_acc_val, label = 'validation accuracy')
    plt.legend()
    plt.show()

    model_ft.eval()
    y_label = []
    y_predict = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            images, labels = data
            N = images.size(0)
            images = Variable(images).to(device)
            outputs = model_ft(images)
            prediction = outputs.max(1, keepdim=True)[1]
            y_label.extend(labels.cpu().numpy())
            y_predict.extend(np.squeeze(prediction.cpu().numpy().T))
    # compute the confusion matrix
    confusion_mtx = confusion_matrix(y_label, y_predict)

    plot_labels = [str(i) for i in range(43)]
    plot_confusion_matrix(confusion_mtx, plot_labels)

    # Generate a classification report
    report = classification_report(y_label, y_predict, target_names=plot_labels)
    print(report)


    label_frac_error = 1 - np.diag(confusion_mtx) / np.sum(confusion_mtx, axis=1)
    sns.countplot(x=label_frac_error)
    plt.xlabel('True Label')
    plt.ylabel('Fraction classified incorrectly')
    plt.figure(figsize=(15,7))
