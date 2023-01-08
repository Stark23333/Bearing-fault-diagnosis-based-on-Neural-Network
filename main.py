import scipy.io as scio
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt


# 将设备设置为GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 定义device（GPU）
print('device:', device)

# 数据预处理
dataFile = 'Train_Test.mat'
dataset = scio.loadmat(dataFile)
Train_data = np.array(dataset['Train_data'], dtype=np.float32)
Train_label = np.array(dataset['Train_label'], dtype=np.float32)
Test_data = np.array(dataset['Test_data'], dtype=np.float32)
Test_label = np.array(dataset['Test_label'], dtype=np.float32)

Train_data, Train_label, Test_data, Test_label = map(torch.tensor,
                                                     (Train_data, Train_label, Test_data, Test_label))
if torch.cuda.is_available():
    Train_data = Train_data.to(device)
    Train_label = Train_label.to(device)
    Test_data = Test_data.to(device)
    Test_label = Test_label.to(device)

bs = 300  # batch_size
train_ds = TensorDataset(Train_data, Train_label)
test_ds = TensorDataset(Test_data, Test_label)


def get_data(train_ds, test_ds, bs):
    return(
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(test_ds, batch_size=bs*2, shuffle=True),
    )


# 搭建五层全连接神经网络
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1200, 256)  # 数据集有1200列，故第一层包含1200个神经元
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 4)      # 共4种故障类别，故输出层包含4个神经元

        # 构造Dropout方法，在每次训练过程中都随机掐死百分之二十的神经元，防止过拟合
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.dropout(F.relu(self.fc4(x)))
        x = F.log_softmax(self.fc5(x), dim=1)    # 输出层不进行Dropout

        return x


# 神经网络训练
model = Classifier()
model = model.to(device)    # 将模型加载到GPU

criterion = nn.CrossEntropyLoss()     # 交叉熵损失函数
criterion = criterion.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 20
train_losses, test_losses = [], []

print('开始训练')
for e in range(epochs):
    running_loss = 0
    train_dl, test_dl = get_data(train_ds, test_ds, bs)
    for i, data in enumerate(train_dl, 1):
        x_data, x_label = data
        # print(' batch:{0} x_data:{1}  label: {2}'.format(i, x_data, x_label))
        optimizer.zero_grad()              # 导数置0
        log_ps = model(x_data)             # 正向传播
        loss = criterion(log_ps, x_label)  # 计算损失
        loss.backward()                    # 反向传播
        optimizer.step()                   # 更新权重
        running_loss += loss.item()        # 损失求和

    else:
        test_loss = 0
        accuracy = 0

        with torch.no_grad():            # 测试的时候不需要开自动求导和反向传播
            model.eval()                 # 关闭Dropout

            for i, data in enumerate(test_dl):
                y_data, y_label = data
                log_ps = model(y_data)                               # 正向传播
                test_loss += criterion(log_ps, y_label)              # 损失求和
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1, largest=True)
                top_p_y, top_class_y = y_label.topk(1, dim=1, largest=True)
                equals = top_class == top_class_y

                accuracy += torch.mean(equals.type(torch.FloatTensor))
        # 恢复Dropout
        model.train()
        # 将训练误差和测试误差存在两个列表里，后面绘制误差变化折线图用
        train_losses.append(running_loss / len(train_dl))
        test_losses.append(test_loss / len(test_dl))

        print("训练集学习次数: {}/{}.. ".format(e + 1, epochs),
              "训练误差: {:.3f}.. ".format(running_loss / len(train_dl)),
              "测试误差: {:.3f}.. ".format(test_loss / len(test_dl)),
              "模型分类准确率: {:.3f}".format(accuracy / len(test_dl)))


# 模型评估
plt.plot(torch.tensor(train_losses, device='cpu'), label='Training loss')
plt.plot(torch.tensor(test_losses, device='cpu'), label='Validation loss')
plt.legend()
plt.show()



