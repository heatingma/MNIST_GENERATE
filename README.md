说明文档

train_func.py 文件用于训练模型参数

module.py 文件是我们的神经网络模型

utils.py 文件是一些函数，主要用于获得一些输出图像

aigcmn.py 文件中的接口类AiGcMn实现了输入n维tensor（n是batch的大小，每个整数在0~9范围内，代表需要生成的数字），输出n*1*28*28的tensor

AiGcMn 有2个生成函数，分别为generate和generate2,2个生成函数采用了不同的训练模型与训练参数。

注：optimize是优化函数，可以使输出的tensor更接近MNIST数据集

data文件夹 是MNIST原始数据集

weights文件夹 存放了预训练参数

example文件夹 存放了我们训练时获得的一些图片信息

使用样例

```python
from aigcmn import AiGcMn
import torch
ai = AiGcMn()
label = torch.randint(low=0,high=10,size=(1,100)).squeeze()
for i in range(10):
    print(label[i*10:i*10+10])

ai.generate(label)
#ai.generate2(label)

```