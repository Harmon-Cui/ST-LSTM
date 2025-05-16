import torch
from matplotlib import pyplot as plt
import numpy as np
from IPython import display
import random
import torchvision
import time
import torchvision.transforms as transforms
import pandas as pd
import datetime
import math
import torch.nn as nn
from sklearn.metrics import mean_squared_error,mean_absolute_error
import timeit

# 定义主体程序
def process(cir_num,od_predict, choice_num, w_1, w_2, w_3, device, train_seq_proportion, window_size, FEATURE_SIZE, OUTPUT_SIZE, num_layers, hidden_size, lr, epochs, batch_size):
    # 提取除待预测车站外，其余所有车站的特征值（空间特征）
    other_pearson = data_pearson[[od_predict, od_predict + 'pearson']]
    other_od = np.array(data_pearson[od_predict]).tolist()
    pearson_list, flow_list, distance_list = [], [], []
    print('开始循环，将其他车站的特征参数存入列表中:' + str(time.asctime(time.localtime(time.time()))))
    # 将所有其他OD对的特征参数都输入列表中，以便后续转为张量输入前馈神经网络
    for i in range(len(other_od)):
        od_i = other_od[i]
        # 根据OD名称寻找相应的特征值
        od_pearson = data_pearson[data_pearson[od_predict] == od_i][od_predict + 'pearson'].values[0]
        pearson_list.append(od_pearson)
        od_flow = data_flow[data_flow[od_predict] == od_i][od_predict + ' ' + 'flow coefficient'].values[0]
        flow_list.append(od_flow)
        od_distance = data_distance.loc[
            (data_distance['O_station_name'] == od_predict) &
            (data_distance['D_station_name'] == od_i),
            'Standardized distance coefficient'
        ].values
        od_distance = od_distance[0] if len(od_distance) > 0 else 0
        distance_list.append(od_distance)
    print('特征提取完毕:' + str(time.asctime(time.localtime(time.time()))))

    # 划分训练集和测试集的函数
    def data_split(train_set_proportion, data_seq):
        data_len = len(data_seq)
        train_set_len = int(data_len * train_set_proportion)
        train_set = data_seq[: train_set_len]
        test_set = data_seq[train_set_len:]
        return train_set, test_set

    # 数据标准化函数（最大-最小标准化）
    def normalize(data):
        min = np.amin(data)
        max = np.amax(data)
        return (data - min) / (max - min)

    # 逆向还原标准化后的数据
    def unnormalize(data_seq, data_set):
        min = np.amin(data_set)
        max = np.amax(data_set)
        return data_seq * (max - min) + min

    # 采样短序列函数
    def create_dataset(data, n_predictions, n_next):
        dim = data.shape[1]
        train_X, train_Y = [], []
        for i in range(data.shape[0] - n_predictions - n_next + 1):
            a = data[i:(i + n_predictions), :]
            train_X.append(a)
            tempb = data[(i + n_predictions):(i + n_predictions + n_next), :]
            b = []
            for j in range(len(tempb)):
                b.append(tempb[j, 0])
            train_Y.append(b)
        train_X = np.array(train_X, dtype='float64')
        train_Y = np.array(train_Y, dtype='float64')
        return train_X, train_Y

    # 利用torch.nn.LSTM构建LSTM模型
    class LSTM(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, num_layers):
            super(LSTM, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
            self.reg = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            x, _ = self.lstm(x)
            return self.reg(x)

    def batch_list(data, batch_size):
        data_length = data.shape[0]
        batch_list = []
        num_batches = math.ceil(data_length / batch_size)  # 向上取整计算
        for batch_index in range(num_batches):
            start_index = batch_index * batch_size
            end_index = min((batch_index + 1) * batch_size, data_length)
            batch_list.append(data[start_index:end_index])
        return batch_list

    space_cor = []
    print('开始叠加参数输出特征值:' + str(time.asctime(time.localtime(time.time()))))
    # 遍历所有OD输出最后的参数
    for i in range(len(other_od)):
        space_parameter = w_1 * pearson_list[i] + w_2 * flow_list[i] + w_3 * distance_list[i]
        space_cor.append(space_parameter)
    data_space_cor = pd.DataFrame({'OD': other_od, 'space_cor': space_cor})
    # print(data_space_cor[data_space_cor.isnull().values == True])
    print('得到空间相关性参数并排序:' + str(time.asctime(time.localtime(time.time()))))
    data_space_sort = data_space_cor.sort_values(by=['space_cor'], ascending=False, inplace=False)  # 降序为False
    data_space_sort = data_space_sort.reset_index(drop=True)
    space_od = np.array(data_space_sort.loc[:choice_num - 1, 'OD']).tolist()
    print(space_od)
    # 用提取出的OD对的时间序列数据、待预测OD本身数据以及待预测站点对应的进出站时序数据为总数据集
    all_select = [od_predict] + space_od
    data_select = data_weekday[all_select]
    # 待输入的OD序列（进行标准化）
    seq_predict = np.array(data_select[od_predict]).tolist()
    od1_select = np.array(data_select[space_od[0]]).tolist()
    od2_select = np.array(data_select[space_od[1]]).tolist()
    od3_select = np.array(data_select[space_od[2]]).tolist()

    # 长序列划分（训练集与测试集的划分）
    train_1, test_1 = data_split(train_seq_proportion, seq_predict)
    train_2, test_2 = data_split(train_seq_proportion, od1_select)
    train_3, test_3 = data_split(train_seq_proportion, od2_select)
    train_4, test_4 = data_split(train_seq_proportion, od3_select)

    # 将训练集和测试集组成[样本数,时间步长,特征数]的形式
    train_seq = torch.tensor(np.array([train_1, train_2, train_3, train_4])).t().numpy()
    test_seq = torch.tensor(np.array([test_1, test_2, test_3, test_4])).t().numpy()
    # train_seq = torch.tensor(np.array([train_1, train_5, train_6])).t().numpy()
    # test_seq = torch.tensor(np.array([test_1, test_5, test_6])).t().numpy()
    # 训练集与测试集按时间步划分
    train_set_1, train_label_1 = create_dataset(train_seq, window_size, 1)
    test_set_1, test_label_1 = create_dataset(train_seq, window_size, 1)
    train_set = np.array(normalize(train_set_1))
    train_label = np.array(normalize(train_label_1))
    test_set = np.array(normalize(test_set_1))
    test_label = np.array(normalize(test_label_1))
    print(train_set.shape)
    print(test_set.shape)
    # 初始化模型，定义loss函数和优化器
    model = LSTM(FEATURE_SIZE, hidden_size, OUTPUT_SIZE, num_layers).to(device)
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print("开始训练模型：" + str(time.asctime(time.localtime(time.time()))))
    # 训练函数
    train_ls = []
    test_ls = []
    score_log = []
    rmse_score_list = []
    mae_score_list = []
    mape_score_list = []

    train_batch = batch_list(train_set, batch_size)
    train_label_batch = batch_list(train_label, batch_size)
    test_batch = batch_list(test_set, batch_size)
    test_label_batch = batch_list(test_label, batch_size)

    m = 0
    for epoch in range(epochs):
        all_prediction_1 = []
        for i in range(len(train_batch)):
            x = torch.from_numpy(train_batch[i]).float().to(device)
            label = torch.from_numpy(train_label_batch[i]).float().to(device)
            out = model(x)  # 给x降维，降成1维
            prediction = out[:, -1, :].squeeze(-1)  # 得到预测值并降维
            loss = loss_func(prediction, label.squeeze(-1))  # 计算损失函数
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 求损失函数对w和b的梯度
            optimizer.step()
            ls = loss.detach().cpu().numpy().tolist()
            train_ls.append(ls)
            prediction_1 = unnormalize(prediction, np.array(data_select[od_predict]).tolist())  # 将预测值逆标准化还原成原值
            all_prediction_1.append(prediction_1.detach().cpu().numpy().tolist())
        all_prediction_1 = np.concatenate(all_prediction_1)  # 拼接函数
        all_prediction_1 = np.array(all_prediction_1)
        all_prediction_1 = np.squeeze(all_prediction_1)
        all_label_1 = train_label_1.squeeze(-1)
        # 测试函数
        all_prediction_2= []
        for i in range(len(test_batch)):
            x = torch.from_numpy(test_batch[i]).float().to(device)
            test_label = torch.from_numpy(test_label_batch[i]).float().to(device)
            out = model(x)
            prediction = out[:, -1, :].squeeze(-1)  # 得到预测值，并降维
            loss = loss_func(prediction, test_label.squeeze(-1))
            lst = loss.detach().cpu().numpy().tolist()
            test_ls.append(lst)
            prediction = unnormalize(prediction, np.array(data_select[od_predict]).tolist())  # 将预测值逆标准化还原成原值
            all_prediction_2.append(prediction.detach().cpu().numpy().tolist())
        all_prediction_2 = np.concatenate(all_prediction_2)  # 拼接函数
        all_prediction_2 = np.array(all_prediction_2)
        all_prediction_2 = np.squeeze(all_prediction_2)
        all_label_2 = test_label_1.squeeze(-1)
        print('第'+ str(cir_num) +'次大循环，完成第' + str(m) + '次迭代:' + str(time.asctime(time.localtime(time.time()))))
        rmse_score = math.sqrt(mean_squared_error(all_label_2, all_prediction_2))
        mae_score = mean_absolute_error(all_label_2, all_prediction_2)
        rmse_score_list.append(rmse_score)
        mae_score_list.append(mae_score)
        m = m + 1
        if epoch == epochs - 1:
            print('第'+ str(cir_num) +'次大循环，最终迭代，RMSE: %.4f, MAE: %.4f' % (rmse_score, mae_score))
            rmse_cir = rmse_score
            mae_cir = mae_score
            all_label = all_label_1 + all_label_2
            all_prediction = all_prediction_1 + all_prediction_2
    return rmse_cir, mae_cir, all_label, all_prediction

# 程序入口（输入变量）
# 待预测OD对名称
choice_num = 3 # 提取空间相关性较强的前choice_num个OD对的数据
# 设置空间相关性指标参数
w_1 = 1
w_2 = 1
w_3 = 0.8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 训练集与测试集划分的比例
train_seq_proportion = 29/31
# 数据划分时间步长
window_size = 96  # 用前24个时间步数据预测第25个时间步的数据
# 模型参数
FEATURE_SIZE = choice_num + 1  # 输入层神经元个数
OUTPUT_SIZE = 1  # 输出层神经元个数
num_layers = 2  # 循环神经网络隐藏层个数
hidden_size = 64  # 隐藏层神经元个数
lr=0.0005  # 学习率
epochs = 500  # 迭代次数
batch_size = 12  # 每次读取的批量个数

start = timeit.default_timer()
print('开始读取空间相关性数据:' + str(time.asctime(time.localtime(time.time()))))
# 读取待输入的特征值
data_pearson = pd.read_csv('Pearson Correlation Between Stations (Example).csv')
data_flow = pd.read_csv('Flow Coefficient Between Stations (Example).csv')
data_distance = pd.read_csv('Distance coefficient between stations (Example).csv')
print('读取完毕:' + str(time.asctime(time.localtime(time.time()))))
print("读取出站量历史数据：" + str(time.asctime(time.localtime(time.time()))))
data_read = pd.read_csv('Sample data of outbound passenger flow.csv')
data_read['time_series'] = pd.to_datetime(data_read['time_series'])
data_read['星期'] = data_read['time_series'].dt.weekday  # 提取日期索引，0是周一，6是周日
data_read['日期'] = data_read['time_series'].dt.day
# 提取工作日的数据，并剔除了南京数据20 21 22三天异常数据
data_weekday = data_read[(data_read['星期'].isin([0, 1, 2, 3, 4])) & ~(data_read['日期'].isin([20, 21, 22]))]
data_weekday.reset_index(drop=True)

print("完成处理历史数据：" +str(time.asctime(time.localtime(time.time()))))
# 提取待预测的OD名称
od_df = pd.read_csv('Stations to be predicted (Example).csv')
od_pair = np.array(od_df['Station to be predicted']).tolist()
od_list = []
rmse_list = []
mae_list = []
value_list = []
for i in range(len(od_pair)):
    od_predict = od_pair[i]
    cir_num = i + 1
    od_list.append(od_predict)
    rmse_cir, mae_cir, all_label, all_prediction = process(cir_num, od_predict, choice_num, w_1, w_2, w_3, device, train_seq_proportion, window_size, FEATURE_SIZE, OUTPUT_SIZE, num_layers, hidden_size, lr, epochs, batch_size)
    rmse_list.append(rmse_cir)
    mae_list.append(mae_cir)
    value_list.append(all_label)
    value_list.append(abs(all_prediction))
print("开始建立表格：" + str(time.asctime(time.localtime(time.time()))))
df_out = pd.DataFrame({'station_name': od_list, 'RMSE': rmse_list, 'MAE': mae_list})
rmse_mean = df_out['RMSE'].mean()
mae_mean = df_out['MAE'].mean()
df_out.loc[0]=[ 'Average value of all stations', rmse_mean, mae_mean]
name_list_1 = []
for i in range(len(od_list)):
    name_list_1.append(str(od_list[i]) + 'true')
    name_list_1.append(str(od_list[i]) + 'predicted')
df_out_1 = pd.DataFrame(columns = name_list_1)
for i in range(len(name_list_1)):
    df_out_1[name_list_1[i]] = value_list[i]

print("开始写入最终csv,当前时间：" + str(time.asctime(time.localtime(time.time()))))
df_out.to_csv("Prediction accuracy of outbound volume.csv" , index = False, date_format = '%Y-%m-%d-%H:%M:%S', encoding="utf_8_sig")
df_out_1.to_csv("Outbound volume prediction value.csv" , index = False, date_format = '%Y-%m-%d-%H:%M:%S', encoding="utf_8_sig")
print("结束,当前时间：" + str(time.asctime(time.localtime(time.time()))))
end_last = timeit.default_timer()
print('run time is: ', str(end_last - start), ' s')









