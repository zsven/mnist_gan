# mnist_gan

GAN(Generative Adversarial Nets)是Ian Goodfellow在2014年发表的论文中提出，受到业界大牛Lecun的推崇，成为机器学习领域的新宠。

## 相关论文
1. [Generative Adversarial Networks](https://arxiv.org/pdf/1406.2661.pdf)
2. [Conditional Generative Adversarial Nets](https://arxiv.org/pdf/1411.1784.pdf)
3. [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434.pdf)

## 原理

#### Generative Adversarial Networks
GAN的基本原理启发自二人零和博弈，博弈方分别由生成模型G(Generator)和判别模型D(Discriminator)充当。它们的功能分别是：
+ G是生成网络，它接受一个带噪声的数据集z，生成模型G(z)
+ D是判别网络，判别一张图片是不是真实的：如果样本来源于真实数据，输出为1，如果样本来源于G，输出为0.
+ 交替迭代训练，G和D都优化自己的网络，形成对抗。直到双方达到一个平衡，即生成模型G生成的样本判别模型判别不出结果了，准确率为50%。

上述公式可用如下公式来表示：
```math
min_{G}max_{D}V(D,G)=E_{x~p_{data}(x)}[logD(x)]+E_{z~p_{z}(z)}[log(1-D(G(z)))]
```

#### Conditional Generative Adversarial Nets
与论文1相比，多了一个Conditional，即给GAN加了一个条件y，让生成的样本符合我们的预期。用公式表示为:
```math
min_{G}max_{D}V(D,G)=E_{x~p_{data}(x)}[logD(x|y)]+E_{z~p_{z}(z)}[log(1-D(G(z|y)))]
```

#### DCGAN
论文3，简称DCGAN，将CNN和GAN结合 ，是GAN在卷积神经网络上最好的尝试。DCGAN对卷积神经网络的结构做了一些改变：
1. 取消了pooling层，D层使用stride来代替pooling。G使用转置卷积进行向上采样；
2. 在D和G中均使用batch normalization
3. 去掉FC层，使网络变为全卷积网络
4. G使用Relu最为激活函数，D使用LeakyRelu最为激活函数。