# mnist_gan

GAN(Generative Adversarial Nets)��Ian Goodfellow��2014�귢���������������ܵ�ҵ���ţLecun���Ƴ磬��Ϊ����ѧϰ������³衣

## �������
1. [Generative Adversarial Networks](https://arxiv.org/pdf/1406.2661.pdf)
2. [Conditional Generative Adversarial Nets](https://arxiv.org/pdf/1411.1784.pdf)
3. [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434.pdf)

## ԭ��

#### Generative Adversarial Networks
GAN�Ļ���ԭ�������Զ�����Ͳ��ģ����ķ��ֱ�������ģ��G(Generator)���б�ģ��D(Discriminator)�䵱�����ǵĹ��ֱܷ��ǣ�
+ G���������磬������һ�������������ݼ�z������ģ��G(z)
+ D���б����磬�б�һ��ͼƬ�ǲ�����ʵ�ģ����������Դ����ʵ���ݣ����Ϊ1�����������Դ��G�����Ϊ0.
+ �������ѵ����G��D���Ż��Լ������磬�γɶԿ���ֱ��˫���ﵽһ��ƽ�⣬������ģ��G���ɵ������б�ģ���б𲻳�����ˣ�׼ȷ��Ϊ50%��

������ʽ�������¹�ʽ����ʾ��
```math
min_{G}max_{D}V(D,G)=E_{x~p_{data}(x)}[logD(x)]+E_{z~p_{z}(z)}[log(1-D(G(z)))]
```

#### Conditional Generative Adversarial Nets
������1��ȣ�����һ��Conditional������GAN����һ������y�������ɵ������������ǵ�Ԥ�ڡ��ù�ʽ��ʾΪ:
```math
min_{G}max_{D}V(D,G)=E_{x~p_{data}(x)}[logD(x|y)]+E_{z~p_{z}(z)}[log(1-D(G(z|y)))]
```

#### DCGAN
����3�����DCGAN����CNN��GAN��� ����GAN�ھ������������õĳ��ԡ�DCGAN�Ծ��������Ľṹ����һЩ�ı䣺
1. ȡ����pooling�㣬D��ʹ��stride������pooling��Gʹ��ת�þ���������ϲ�����
2. ��D��G�о�ʹ��batch normalization
3. ȥ��FC�㣬ʹ�����Ϊȫ�������
4. Gʹ��Relu��Ϊ�������Dʹ��LeakyRelu��Ϊ�������