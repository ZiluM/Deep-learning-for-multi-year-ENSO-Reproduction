"""
TrainFuncVal.py
此脚本写用于训练的函数
"""
import torch
import datetime
import scipy.stats as sps
from DataLoaderFunc import ENSODataset
from torch.utils.data import DataLoader
import pickle
from CNNClass import ConvNetwork


def valFunc(Network, DataL=iter(DataLoader(ENSODataset(type_="OBSVal"), batch_size=1000, shuffle=False)),
            criterion=torch.nn.MSELoss()):
    """
    对神经网络训练的结果进行验证，使用文中给的相关系数和Loss进行验证
    Network:需要验证的神经网络
    DataL: 需要验证的神经网络数据集加载器，默认为 OBSVal,DataLoader需要一次Load完响相应数据
    criterion: 判别器，默认为MSELoss
    return
    MSELoss , 每个月相关系数 ， 每个相关系数的 P值
    """
    inputs, outputs = next(DataL)
    # 加载到CPU验证
    Network.to("cpu")
    Network.eval()
    pred = Network.forward(inputs)
    LossVal = criterion(pred, outputs).item()
    Calpred = pred.T.detach().numpy()
    CalOutputs = outputs.T.detach().numpy()
    ACCList = []
    PList = []
    # 对23个月计算相关系数
    for index_month in range(23):
        acc, p_value = sps.pearsonr(Calpred[index_month], CalOutputs[index_month])
        ACCList.append(acc)
        PList.append(p_value)
    return LossVal, ACCList, PList


def trainFunc(Network, DataL, epochs, optim, saveName, criterion=torch.nn.MSELoss(), device=torch.device("cpu"),
              ValMethod=True, gen_log=True, Save=True,
              val_data=iter(DataLoader(ENSODataset(type_="OBSVal"), batch_size=1000, shuffle=False))):
    """
    此函数对神经网络进行训练、保存
    Network:需要验证的神经网络
    DataL: 需要训练神经网络数据集加载器
    epochs :需要训练的回合数
    optim : 神经网络优化器
    saveName : 神经网络保存名称
    criterion : 判别器 默认torch.nn.MSELoss()
    device : 训练的设备 ， 下载了CUDA可以使用GPU训练
    ValMethod : 是否需要验证
    gen_log : 是否需要产生训练日志
    Save ： 神经网络参数是否需要保存
    val_data ： 验证数据集
    """
    Network.to(device)  # 加载网路到设备
    Network.train()  # 开始训练
    lossList = []  # 用于保存训练中的 Loss

    for epoch in range(epochs):
        for AData, (inputs, outputs) in enumerate(DataL):
            # 开始 一个 batch训练
            Network.train()
            # 加载数据到设备
            inputs = inputs.to(device)
            outputs = outputs.to(device)
            # 优化器 清零
            optim.zero_grad()
            # 神经网络输出
            pred = Network.forward(inputs)
            # 计算loss并保存
            loss = criterion(pred, outputs)
            lossList.append(loss.item())
            # 反向传播算法
            loss.backward()
            optim.step()
            # 对一部分训练过程的Loss进行打印
            if AData % 5 == 0:
                print('epoch={},batch={},损失 = {:.6f}'.format(epoch, AData, loss.item()))
    # 获取时间
    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
    # 保存网络参数
    if Save is True:
        torch.save(Network.state_dict(), "./NetParam/%s.net" % saveName)
    # 神经网络进行检验
    if ValMethod is True:
        LossVal, ACCList, PList = valFunc(Network, DataL=val_data)
    else:
        LossVal, ACCList, PList = None, None, None
    # 生成并保存日志
    if gen_log is True:
        SaveDict = {"trainName": saveName, "train_time": time_str, "epoch_num": epochs, "lossList": lossList,
                    "LossVal": LossVal,
                    "ACCList": ACCList, "Plist": PList}
        saveF = open("TrainRes/%s.pickle" % saveName, "wb")
        pickle.dump(SaveDict, saveF)
        saveF.close()


if __name__ == '__main__':
    Net = ConvNetwork(30, 30)
    # # print(valFunc(Net)[0])
    # DS = ENSODataset("CMIP")
    # DL = DataLoader(DS, batch_size=400, shuffle=True)
    # trainFunc(Net, DL, 10, saveName="try", optim=torch.optim.Adam(Net.parameters()))
    # Net.load_state_dict(torch.load("./NetParam/try.net"))
    DS2 = ENSODataset("CMIP", T_begin=1870, T_end=1970)
    DL2 = DataLoader(DS2, batch_size=400, shuffle=True)
    valD = ENSODataset("CMIP", T_begin=1985, T_end=2010)
    valD = iter(DataLoader(valD, batch_size=1000))
    trainFunc(Net, DL2, 20, saveName="try", optim=torch.optim.Adam(Net.parameters()), val_data=valD)
    # trainPlot("TrainRes/try.pickle")
