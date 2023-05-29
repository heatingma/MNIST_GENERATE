from aigcmn import AiGcMn
import torch
ai = AiGcMn()
label = torch.randint(low=0,high=10,size=(1,100)).squeeze()
for i in range(10):
    print(label[i*10:i*10+10])


mode = ['part','all']
pretrain = ['1','2','3']

for i in range(3):
    for j in range(2):
        ai.generate(label,mode=mode[j],show=True,pretrain=pretrain[i])


