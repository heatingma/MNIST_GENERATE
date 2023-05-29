import os
import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim
from tqdm import trange,tqdm
from utils import sample_digits,tensor_to_img
from module import Classifier,Generator,Discriminator,Discriminator_num,Generator_num
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Lambda
from torch.utils.data import DataLoader, RandomSampler

device = 'cuda' if torch.cuda.is_available() else 'cpu'

trans = Compose([ToTensor(), Lambda(lambda x: x * 2 - 1)])
train_set = MNIST('./data', train=True, transform=trans, download=True)
test_set = MNIST('./data', train=False, transform=trans, download=True)

def classifier_train(batch_size=100):
    # gain train_data
    train_num = 60000
    train_sampler = RandomSampler(train_set, replacement=True, num_samples=train_num)
    train_loader = DataLoader(train_set, batch_size, sampler=train_sampler, drop_last=True)
    
    # train 
    train_epochs = trange(int(train_num/batch_size), ascii=True, leave=True, desc="Epoch", position=0)
    classifier = Classifier(11).to(device) 
    classifier.load_state_dict(torch.load("weights/classifier_weights.pt"))
    print("TRAINING",'.'*20)
    classifier.train()
    train_loss_sum = 0
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(params=classifier.parameters(),lr=0.001)

    for epoch in train_epochs:
        cur_datasets = next(iter(train_loader))
        imgs = cur_datasets[0].to(device)
        labels = cur_datasets[1].to(device)    
        for _ in range(int(batch_size/ 10)):
            random_num = random.randint(1,batch_size-2)
            imgs[random_num] = (imgs[random_num] + imgs[random_num+1] + imgs[random_num-1])/3
            labels[random_num] = 10
        
        out = classifier(imgs).to(device)
        opt.zero_grad()
        loss = criterion(out,labels)
        loss.backward()
        train_loss_sum = train_loss_sum + loss.item()
        train_loss_score = train_loss_sum / ((epoch+1)*batch_size)
        opt.step()
        train_epochs.set_description("Epoch (Loss=%g)" % round(train_loss_score, 5))

    torch.save(classifier.state_dict(),"weights/classifier_weights.pt")

def dis_gen_train_real(batch_size, iterations, sample_interval, generator, discriminator, criterion, g_optim, d_optim):

    sampler = RandomSampler(train_set, replacement=True, num_samples=batch_size * iterations)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, drop_last=True)
    y_real = torch.ones(batch_size, 1).to(device)
    y_fake = torch.zeros(batch_size, 1).to(device)

    losses = []
    iteration = 0
    d_loss = 0.0
    g_loss = 0.0

    for x, labels in tqdm(train_loader):
        x = x.to(device)
        labels = labels.to(device)

        # 1. Train discriminator on real images
        d_out = discriminator(x, labels)
        d_loss_real = criterion(d_out, y_real)
        d_optim.zero_grad()
        d_loss_real.backward()
        d_optim.step()

        # 2. Train discriminator on fake images

        x_gen = generator(labels)
        d_out = discriminator(x_gen, labels)
        d_loss_fake = criterion(d_out, y_fake)
        d_optim.zero_grad()
        d_loss_fake.backward()
        d_optim.step()

        # 3. Train generator to force discriminator making false predictions

        x_gen = generator(labels)
        d_out = discriminator(x_gen, labels)
        g_loss_real = criterion(d_out, y_real)
        g_optim.zero_grad()
        g_loss_real.backward()
        g_optim.step()

        # Calculate discriminator and generator loss for visualization
        d_loss += 0.5 * (d_loss_real + d_loss_fake).item()
        g_loss += g_loss_real.item()
        iteration += 1
        
        if iteration % sample_interval == 0:
            # Calculate the loss for the last sample_interval iterations
            d_loss = d_loss / sample_interval
            g_loss = g_loss / sample_interval
            
            print(f"D-Loss: {d_loss:.4f} G-Loss: {g_loss:.4f}")
            
            losses.append((d_loss, g_loss))
            d_loss = 0
            g_loss = 0

    return losses

def dis_gen_train(epoch_num = 3):
    for epoch in range(epoch_num):
        generator = Generator(10).to(device)
        if os.path.exists("weights/generator_weights.pt"):
            print("using pretrained generator weights")
            generator.load_state_dict(torch.load("weights/generator_weights.pt"))

        discriminator = Discriminator(10).to(device)
        if os.path.exists("weights/discriminator_weights.pt"):
            print("using pretrained discriminator weights")
            discriminator .load_state_dict(torch.load("weights/discriminator_weights.pt"))

        g_optim = optim.Adam(generator.parameters(), lr=0.001)
        d_optim = optim.Adam(discriminator.parameters(), lr=0.001)
        criterion = nn.BCELoss()

        losses = dis_gen_train_real(16,2000,100,generator,discriminator,criterion,g_optim,d_optim)

        sample_digits(generator,epoch=epoch)
        
        save = input("save or not?  y/n")
        if save == 'y':
            torch.save(generator.state_dict(),"weights/generator_weights.pt")
            torch.save(discriminator.state_dict(),"weights/discriminator_weights.pt")

def get_label_data(num = 0,batch_size = 10):
    x = train_set.data / 255 * 2 - 1
    x = x.unsqueeze(dim=1)
    label = train_set.targets
    data = list()
    part = list()
    part_num = 0
    for i in range(len(label)):
        if label[i] == num:
            part_num += 1
            part.append(x[i].numpy())
            if part_num % 10 == 0:
                data.append(torch.tensor(np.array(part)))
                part_num = 0
                part = list()
    return data

def number_train_real(num=0, batch_size=10):
    gen = Generator_num().to(device)
    dis = Discriminator_num().to(device)
    gen_path = "weights/GEN_{}.pt".format(num)
    dis_path = "weights/DIS_{}.pt".format(num)
    if os.path.exists(gen_path):
        gen.load_state_dict(torch.load(gen_path))
    if os.path.exists(dis_path):
        dis.load_state_dict(torch.load(dis_path))

    data = get_label_data(num)
    epochs = trange(len(data), ascii=True, leave=True, desc="Epoch", position=0)
    criterion = nn.BCELoss()
    d_optim = optim.Adam(dis.parameters(), lr=0.001)
    g_optim = optim.Adam(gen.parameters(), lr=0.001)
    d_loss = 0
    g_loss = 0
    y_real = torch.ones(batch_size,1).to(device)
    y_fake = torch.zeros(batch_size,1).to(device)
    for epoch in epochs:
        x = data[epoch].to(device)    
        # 1. Train discriminator on real images
        d_out = dis(x)
        d_loss_real = criterion(d_out,y_real)
        d_optim.zero_grad()
        d_loss_real.backward()
        d_optim.step()

        # 2. Train discriminator on fake images
        x_gen = gen(batch_size)
        d_out = dis(x_gen)
        d_loss_fake = criterion(d_out,y_fake)
        d_optim.zero_grad()
        d_loss_fake.backward()
        d_optim.step()

        # 3. Train generator to force discriminator making false predictionss
        x_gen = gen(batch_size)
        d_out = dis(x_gen)
        g_loss_real = criterion(d_out,y_real)
        g_optim.zero_grad()
        g_loss_real.backward()
        g_optim.step()

        d_loss += 0.5 * (d_loss_real + d_loss_fake).item()
        g_loss += g_loss_real.item()
        message = "DLoss={}, GLoss={}".format(round(d_loss/(epoch+1), 5),round(g_loss/(epoch+1), 5))
        epochs.set_description(message)
    
    tensor_to_img(gen(batch_size=100))
    torch.save(gen.state_dict(),"weights/GEN_{}.pt".format(num))
    torch.save(dis.state_dict(),"weights/DIS_{}.pt".format(num))

def number_train():
    for num in range(10):
        for _ in range(5):
            number_train_real(num)

number_train()