import keras
from keras.models import Model,Sequential #贯序模型
from keras.layers import Input,Dense,Dropout,Activation #引入需要使用的层模型
from keras.optimizers import SGD
from keras.datasets import mnist #使用keras自带的MNIST数据
import numpy as np
import matplotlib.pyplot as plt

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        #创建一个图
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')#plt.plot(x,y)，这个将数据画成曲线
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)#设置网格形式
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')#给x，y轴加注释
        plt.legend(loc="upper right")#设置图例显示位置
        plt.show()

#创建一个实例LossHistory
history = LossHistory()


(x_train, y_train), (x_test, y_test) = mnist.load_data() # 下载mnist数据集
print(x_train.shape,y_train.shape) # 60000张28*28的单通道灰度图
print(x_test.shape,y_test.shape)
x_train = x_train.reshape(60000,784) # 将图片摊平，变成向量
x_test = x_test.reshape(10000,784) # 对测试集进行同样的处理
print(x_train.shape)
print(x_test.shape)
#对数据进行归一化处理
x_train = x_train / 255
x_test = x_test / 255
#对y标签进行处理，5 --> [ 0, 0, 0, 0, 0,1, 0, 0, 0, 0] ,使用keras的utils工具集中的函数可以做到
y_train = keras.utils.to_categorical(y_train,10)
y_test = keras.utils.to_categorical(y_test,10)



model=Sequential() #定义模型采用贯序模型方式
# 输入层
#定义512个神经元，Input_shape因为28x28=784 将二维矩阵换成了一维向量输入
#激活函数使用relu
model.add(Dense(512, activation = 'relu',input_shape=(784,)))
model.add(Dropout(0.2))

#隐藏层
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.2))

#隐藏层
#分成10类
model.add(Dense(10, activation = 'softmax'))

#打印模型概况
model.summary()


#定义SGD参数

# lr 学习率
# momentum 动量
# decay 每次更新后的学习率衰减值
# nesterov 确定是否使用Nesterov动量
sgd=SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)

#定义优化器
#使用刚刚定义的SGD优化器，损失函数使用交叉熵
model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])


# # 训练模型
model.fit(x_train, y_train, batch_size=128, epochs=25,validation_data=(x_test,y_test),callbacks=[history])


# # 输出训练结果
score = model.evaluate(x_test,y_test)
print('')
print("loss:",score[0])
print('')
print("accu:",score[1])

history.loss_plot('epoch')
