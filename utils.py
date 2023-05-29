import torch
import matplotlib.pyplot as plt
from math import sqrt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def tensor_to_img(x:torch.Tensor,save=True,path="output"):
    x_shape  = x.shape
    dim = len(x_shape)
    if dim == 3:
        x = x.squeeze(dim=0)
    if dim == 2:
        line = row = 1
    else:
        row = line = int(sqrt(x_shape[0]))
    x = x.cpu().detach().numpy()
    _, axs = plt.subplots(row,line,figsize=(line, row),sharey=True,sharex=True)
    if line == 1:
        axs.imshow(x[0][0],cmap='gray')
        axs.axis('off')
    else:
        cnt = 0
        for i in range(row):
            for j in range(line):
                axs[i, j].imshow(x[cnt][0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
    if save:      
        plt.savefig("example/{}.png".format(path))
    return axs

def sample_digits(generator, image_grid_rows=6, image_grid_columns=10,epoch=0):

    with torch.no_grad():
        generator.eval()
        gen_imgs = generator(torch.tensor([i for _ in range(image_grid_rows) for i in range(0, 10)], device=device))
        gen_imgs = gen_imgs.cpu().detach().view(-1, 28, 28).numpy()
        generator.train()

    gen_imgs = 0.5 * gen_imgs + 0.5
    
    _, axs = plt.subplots(image_grid_rows,
                            image_grid_columns,
                            figsize=(10, 6),
                            sharey=True,
                            sharex=True)

    cnt = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            axs[i, j].imshow(gen_imgs[cnt, :, :], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1

    plt.savefig("example/output_{}.png".format(epoch))