from aigcmn import AiGcMn
import torch
ai = AiGcMn()
label = torch.randint(low=0,high=10,size=(1,100)).squeeze()
for i in range(10):
    print(label[i*10:i*10+10])

ai.generate(label,mode='all',show=True)
ai.generate(label,mode='part',show=True)
ai.generate2(label,mode='all',show=True)
ai.generate2(label,mode='part',show=True)

