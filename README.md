### 说明文档

    文件说明：

        train_func.py 文件用于训练模型参数
        module.py 文件是我们的神经网络模型
        utils.py 文件是一些函数，主要用于获得一些输出图像
        aigcmn.py 文件中的接口类AiGcMn实现了输入n维tensor
            （n是batch的大小，每个整数在0~9范围内，代表需要生成的数字），输出n*1*28*28的tensor

    文件夹说明
        data文件夹 是MNIST原始数据集
        weights文件夹 存放了预训练参数
        example文件夹 存放了我们训练时获得的一些图片信息

### AiGcMn接口说明
    
    
    
```python
    # AiGcMn 提供generate接口，支持随机输入label生成对应随机的MNIST数字
    # generate 函数原型如下
    generate(self, label:torch.Tensor, retrain=True, mode="all", show=False, pretrain="1")
    """
        label: 输入的标签
        retrain: 是否重新训练，
        mode: 重新训练的模式
        show: 是否展示图片/保存图片
        pretrain: 由于我们设计了3种不同的网络，因此提供3种选择
    """
```

### 使用样例

```python
    from aigcmn import AiGcMn
    import torch
    ai = AiGcMn()
    label = torch.randint(low=0,high=10,size=(1,100)).squeeze()
    ai.generate(label)
```

### 结果展示

<div><center>
<img src=example/1_after_all_optimize.png width=100% height=100% >
<br>
<strong><font face="仿宋" size=2>方法1生成的MNIST数字结果</font>
</strong>
</center></div>