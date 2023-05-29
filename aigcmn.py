import torch
from module import Generator,Classifier,Generator_num
from utils import tensor_to_img

device = 'cpu'
MIN_SCORE = 0.85
MAX_ITER = 10

class AiGcMn():
    def __init__(self):
        self.gen = Generator(10).to(device)
        self.gen_num = Generator_num().to(device)
        self.cf = Classifier(11).to(device)
        self.gen.load_state_dict(torch.load("weights/generator_weights.pt"))
        self.cf.load_state_dict(torch.load("weights/classifier_weights.pt"))
        
    def generate(self,label:torch.Tensor,retrain=True,mode="all",show=False):
        label = label.int().to(device)
        gen_img = self.gen(label)
        
        if show:
            tensor_to_img(gen_img,path="1_before_optimize")
        if retrain:
            if mode == 'part':
                retrain_list = self.get_retrain(gen_img,label)
                gen_img = self.optimize(gen_img,retrain_list)
                if show:
                    tensor_to_img(gen_img,path="1_after_part_optimize")
            elif mode == 'all':
                retrain_list = self.all_retrain(label)
                gen_img = self.optimize(gen_img,retrain_list)
                if show:
                    tensor_to_img(gen_img,path="1_after_all_optimize")
            else:
                raise TypeError(mode + " is not supported!" + " available mode : all / part")    
        return gen_img 
    
    def get_retrain(self,gen_img,label):
        cf_score = torch.sigmoid(self.cf(gen_img))
        retrain  = dict()
        for i in range(10):
            retrain[str(i)] = list()
        for i in range(len(label)):
            if cf_score[i][label[i]] < MIN_SCORE:
                retrain[str(label[i].item())].append(i)
        return retrain

    def all_retrain(self,label):
        retrain  = dict()
        for i in range(10):
            retrain[str(i)] = list()
        for i in range(len(label)):
            retrain[str(label[i].item())].append(i)
        return retrain

    def optimize(self,gen_img,retrain):
        for i in range(10):
            cur_retrain = retrain[str(i)]
            length = len(cur_retrain)
            if length == 0:
                continue
            for j in range(length):
                cur_iter = 0
                while(cur_iter < MAX_ITER):
                    cur_iter += 1
                    new_imgs = self.gen(torch.ones(10,dtype=int,device=device) * i)
                    score = self.cf(new_imgs)[:,i]
                    score_max_index = torch.argmax(score)
                    gen_img[cur_retrain[j]] = new_imgs[score_max_index]
                    if score[score_max_index] > MIN_SCORE:
                        break
        return gen_img
                    
    def generate2(self,label:torch.Tensor,retrain=True,mode="all",show=False):
        label = label.int().to(device)
        gen_img = self.gen_num(len(label))

        index  = dict()
        for i in range(10):
            index[str(i)] = list()
        for i in range(len(label)):
            index[str(label[i].item())].append(i)

        for i in range(10):      
            cur_index = index[str(i)]
            length = len(cur_index)
            if length == 0:
                continue
            self.gen_num.load_state_dict(torch.load("weights/GEN_{}.pt".format(i)))
            for j in range(length):
                cur_iter = 0
                while(cur_iter < MAX_ITER):
                    img = self.gen_num(100)
                    score = self.cf(img)[:,i]
                    max_score_index = torch.argmax(score)
                    gen_img[cur_index[j]] = img[max_score_index]
                    if score[max_score_index] > MIN_SCORE:
                        break
        
        if show:
            tensor_to_img(gen_img,path="2_before_optimize")
        
        if retrain:
            if mode == 'part':
                retrain_list = self.get_retrain(gen_img,label)
                gen_img = self.optimize(gen_img,retrain_list)
                if show:
                    tensor_to_img(gen_img,path="2_after_part_optimize")
            elif mode == 'all':
                retrain_list = self.all_retrain(label)
                gen_img = self.optimize(gen_img,retrain_list)
                if show:
                    tensor_to_img(gen_img,path="2_after_all_optimize")
            else:
                raise TypeError(mode + " is not supported!" + " available mode : all / part")  
            
        return gen_img 
                    
        