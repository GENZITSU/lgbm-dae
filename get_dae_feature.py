import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import QuantileTransformer


class TableData(Dataset):
    def __init__(self, data):
        self.data = np.array(data)
        self.data_num = self.data.shape[0]

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        return self.data[idx]


class AutoEncoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(AutoEncoder, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, output_size)
        self.fc2 = nn.Linear(output_size, output_size)
        self.fc3 = nn.Linear(output_size, output_size) # 取得する中間表現
        self.fc4 = nn.Linear(output_size, output_size)
        self.fc5 = nn.Linear(output_size, input_size)
        # 初期化
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)
        nn.init.kaiming_normal_(self.fc4.weight)
        nn.init.kaiming_normal_(self.fc5.weight)

    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        h = self.relu(self.fc3(h))
        h = self.relu(self.fc4(h))
        out = self.fc5(h)
        return out

    def get_representation(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        h = self.relu(self.fc3(h))
        return h


def train(model, data_loader, loss_func, optimizer, device=torch.device("cpu")):
    model.train()
    running_loss = 0
    for row_data in data_loader:
        optimizer.zero_grad()
        row_data = row_data.to(device)
        outputs = model(row_data)
        loss = loss_func(outputs, row_data)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    train_loss = running_loss / len(data_loader)
    return train_loss

def swap_noise(array, noise_level=0.2):
    '''(i, j)要素を確率的に(i', j)に変えるノイズ
    '''
    n_row, n_col = array.shape
    rands = np.random.uniform(0, 1, size=(n_row, n_col))
    copy_array = np.array(array)
    for col in range(n_col):
        for row in range(n_row):
            if rands[row, col] < noise_level:
                swap_target_row = np.random.randint(0, n_row)
                copy_array[row, col] = array[swap_target_row, col]
    return copy_array


def preprocess(merged_df, num_cols, cat_cols,
               n_quantailes=100, random_state=2019, noise_level=0.2):
    '''dae用の前処理を行う
    merged_df: train, testを指し示すfぁっが必要
    '''
    # flagを一旦除外
    train_flag = merged_df["train"].values
    merged_df = merged_df.drop(["train"], axis=1)

    # rankgauss
    rankgauss_transformer = QuantileTransformer(n_quantiles=n_quantailes,
                                                random_state=random_state,
                                                output_distribution="normal")
    merged_df[num_cols] = rankgauss_transformer.\
                                fit_transform(merged_df[num_cols])
    # one hot
    noised_merged_df = pd.DataFrame(swap_noise(merged_df.values, noise_level),
                                    columns=merged_df.columns)
    one_hot_noised_merged_df = pd.get_dummies(noised_merged_df,
                                        columns=cat_cols)

    # 特徴量抽出用のdf
    merged_df["train"] = train_flag
    one_hot_merged_df = pd.get_dummies(merged_df, columns=cat_cols)

    return one_hot_noised_merged_df, one_hot_merged_df


def train_dae(noised_df, device_name, save_path,
              cycle = 300, output_size=100, batch_size=128, learning_rate=1e-3):
    '''dae用のモデルを訓練する
    '''
    # loader
    dataset = TableData(noised_df.values.astype("float32"))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # gpu/cpu
    device = torch.device(device_name)

    # モデル
    input_size = noised_df.values.shape[1]
    dae_model = AutoEncoder(input_size, output_size)
    dae_model = dae_model.to(device)

    #Loss, Optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(dae_model.parameters(),
                                 lr=learning_rate)

    # train
    term = cycle / 10
    for i in range(cycle):
        if (i+1) % term == 0:
            print(train(dae_model, loader, criterion, optimizer, device))
    dae_model.to(torch.device("cpu"))
    torch.save(dae_model.state_dict(), save_path)

    return


def train_dae(noised_df, device_name, save_path,
              cycle = 300, output_size=100, batch_size=128, learning_rate=1e-3):
    '''dae用のモデルを訓練する
    '''
    # loader
    dataset = TableData(noised_df.values.astype("float32"))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # gpu/cpu
    device = torch.device(device_name)

    # モデル
    input_size = noised_df.values.shape[1]
    dae_model = AutoEncoder(input_size, output_size)
    dae_model = dae_model.to(device)

    #Loss, Optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(dae_model.parameters(),
                                 lr=learning_rate)

    # train
    term = cycle / 10
    for i in range(cycle):
        if (i+1) % term == 0:
            print(train(dae_model, loader, criterion, optimizer, device))
    dae_model.to(torch.device("cpu"))
    torch.save(dae_model.state_dict(), save_path)

    return


def get_representation(array, save_file, output_size=100):
    '''学習ずみdaeから特徴量抽出
    '''
    input_size = array.shape[1]
    dae_model = AutoEncoder(input_size, output_size)
    dae_model.load_state_dict(torch.load(save_file))

    dae_model.eval()
    with torch.no_grad():
        inputs = torch.Tensor(array)
        outputs = dae_model.get_representation(inputs)
    return outputs.numpy()

