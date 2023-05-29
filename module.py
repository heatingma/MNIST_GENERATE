import torch
import torch.nn as nn
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
class Generator(nn.Module):
    def __init__(self,num_classes):
        super(Generator,self).__init__()
        self.embed = nn.Embedding(num_classes, 100)
        self.dense = nn.Linear(100, 7*7*256)
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU())
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU())
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Tanh())
        
    def forward(self,label):
        embedded_label = self.embed(label)
        z = torch.randn(len(label),100).to(device)
        x = embedded_label * z
        x = self.dense(x).view(-1, 256, 7, 7)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class Generator_num(nn.Module):
    def __init__(self):
        super(Generator_num, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(100, 256 * 7 * 7)
            #nn.BatchNorm1d(256 * 7 * 7)
        )
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU())
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU())
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Tanh())

    def forward(self,batch_size):
        x = torch.randn(batch_size,100).to(device)
        x = self.linear(x)
        x = x.view(-1, 256, 7, 7)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class Discriminator(nn.Module):
    def __init__(self,num_classes):
        super(Discriminator,self).__init__()
        self.embed = nn.Embedding(num_classes,28*28)
        self.layer1 = nn.Sequential(
            nn.Conv2d(2, 64, 3, stride=2, padding=1),
            nn.LeakyReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU())
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 3 * 128, 1),
            nn.Sigmoid())
        
    def forward(self, x, label):
        embedded_label = self.embed(label).view_as(x)
        x = torch.cat([x, embedded_label], dim=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.dense(x)
        return x

class Discriminator_num(nn.Module):
    def __init__(self):
        super(Discriminator_num,self).__init__()
        self.weight = nn.Parameter(torch.randn(1,1,28,28))
        self.layer1 = nn.Sequential(
            nn.Conv2d(2, 64, 3, stride=2, padding=1),
            nn.LeakyReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU())
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 3 * 128, 1),
            nn.Sigmoid())
        
    def forward(self,x):
        x = torch.cat([x,self.weight.repeat(x.shape[0],1,1,1).to(device)], dim=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.dense(x)
        return x

class Classifier(nn.Module):
    def __init__(self,num_class):
        super(Classifier,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=1),
            nn.LeakyReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU())
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 3 * 128, num_class))
    
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.dense(x)
        return x 

