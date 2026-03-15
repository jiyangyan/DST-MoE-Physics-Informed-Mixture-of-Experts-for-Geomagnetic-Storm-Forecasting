# -*- coding: utf-8 -*-

# 忽略warning输出
import warnings
warnings.filterwarnings('ignore')


# ========== 数据准备与处理部分 ==========

# period名称映射
mapping = {
    'train_a': 'period_1',
    'train_b': 'period_2',
    'train_c': 'period_3',
    'test_a': 'period_4',
    'test_b': 'period_5',
    'test_c': 'period_6',
}

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置matplotlib的backend为Agg
plt.switch_backend('Agg')

# 读取数据并设置索引（public数据）
print("开始数据读取...")

DATA_PATH = Path("public/")
dst = pd.read_csv(DATA_PATH / "dst_labels.csv")
dst['period'] = dst['period'].map(mapping)
dst.timedelta = pd.to_timedelta(dst.timedelta)
dst.set_index(["period", "timedelta"], inplace=True)

sunspots = pd.read_csv(DATA_PATH / "sunspots.csv")
sunspots['period'] = sunspots['period'].map(mapping)
sunspots.timedelta = pd.to_timedelta(sunspots.timedelta)
sunspots.set_index(["period", "timedelta"], inplace=True)

solar_wind = pd.read_csv(DATA_PATH / "solar_wind.csv")
solar_wind['period'] = solar_wind['period'].map(mapping)
solar_wind.timedelta = pd.to_timedelta(solar_wind.timedelta)
solar_wind.set_index(["period", "timedelta"], inplace=True)

satellite_positions = pd.read_csv(DATA_PATH / "satellite_positions.csv")
satellite_positions['period'] = satellite_positions['period'].map(mapping)
satellite_positions.timedelta = pd.to_timedelta(satellite_positions.timedelta)
satellite_positions.set_index(["period", "timedelta"], inplace=True)

# 读取数据并设置索引（private数据）
DATA_PATH2 = Path("private/")

dst2 = pd.read_csv(DATA_PATH2 / "dst_labels.csv")
dst2['period'] = dst2['period'].map(mapping)
dst2.timedelta = pd.to_timedelta(dst2.timedelta)
dst2.set_index(["period", "timedelta"], inplace=True)

sunspots2 = pd.read_csv(DATA_PATH2 / "sunspots.csv")
sunspots2['period'] = sunspots2['period'].map(mapping)
sunspots2.timedelta = pd.to_timedelta(sunspots2.timedelta)
sunspots2.set_index(["period", "timedelta"], inplace=True)

solar_wind2 = pd.read_csv(DATA_PATH2 / "solar_wind.csv")
solar_wind2['period'] = solar_wind2['period'].map(mapping)
solar_wind2.timedelta = pd.to_timedelta(solar_wind2.timedelta)
solar_wind2.set_index(["period", "timedelta"], inplace=True)

satellite_positions2 = pd.read_csv(DATA_PATH2 / "satellite_positions.csv")
satellite_positions2['period'] = satellite_positions2['period'].map(mapping)
satellite_positions2.timedelta = pd.to_timedelta(satellite_positions2.timedelta)
satellite_positions2.set_index(["period", "timedelta"], inplace=True)

# 丢掉source列
solar_wind.drop('source',axis=1, inplace=True)
solar_wind2.drop('source',axis=1, inplace=True)

print("数据读取完成！\n")

# 特征处理
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# 随机种子与可重复性设置
import torch
torch.manual_seed(123)
np.random.seed(2026)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("开始数据处理...")

# 太阳风特征子集合
SOLAR_WIND_FEATURES = [
    "bt",
    "temperature",
    "bx_gse",
    "by_gse",
    "bz_gse",
    "phi_gse",
    "theta_gse",
    "bx_gsm",
    "by_gsm",
    "bz_gsm",
    "phi_gsm",
    "theta_gsm",
    "speed",
    "density",
]

# 训练时会用到的特征
XCOLS = (
    [col + "_mean" for col in SOLAR_WIND_FEATURES]
    + [col + "_std" for col in SOLAR_WIND_FEATURES]
    + ["smoothed_ssn"]
)

def impute_features(feature_df, imp=None):
    """采用以下方法填补缺失数据:
    - `smoothed_ssn`: 前向填充（forward fill）
    - `solar_wind`: 众数填充（interpolation）
    """
    # 前向填充太阳黑子数据
    feature_df.smoothed_ssn = feature_df.smoothed_ssn.fillna(method="ffill")
    # 用众数填充策略填补缺失的太阳风特征
    feature_df = feature_df.reset_index()
    cols = feature_df.columns[2:]
    if imp is None:
        imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        imp.fit(feature_df[cols])
    temp = imp.transform(feature_df[cols])
    feature_df[cols] = temp
    feature_df.timedelta = pd.to_timedelta(feature_df.timedelta)
    feature_df.set_index(["period", "timedelta"], inplace=True)
    return feature_df, imp


def aggregate_hourly(feature_df, aggs=["mean", "std"]):
    """用均值和标准差将特征进行小时级聚合
    e.g. 从 "11:00:00" 到 "11:59:00" 的所有值会被聚合到 "11:00:00".
    """
    # 使用timedelta参数
    agged = feature_df.groupby(
        ["period", feature_df.index.get_level_values(1).floor("H")]
    ).agg(aggs)
    # 展平
    agged.columns = ["_".join(x) for x in agged.columns]
    return agged

def preprocess_features(solar_wind, sunspots, scaler=None, subset=None):
    """
    预处理函数
    """
    if subset:
        solar_wind = solar_wind[subset]

    # 聚合小时数据
    hourly_features = aggregate_hourly(solar_wind).join(sunspots)

    # 标准化
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(hourly_features)

    normalized = pd.DataFrame(
        scaler.transform(hourly_features),
        index=hourly_features.index,
        columns=hourly_features.columns,
    )

    # 填充缺失值
    imputed, imp = impute_features(normalized)

    return imputed, scaler, imp


features, scaler, imputer = preprocess_features(solar_wind, sunspots, subset=SOLAR_WIND_FEATURES)
features2, scaler2, imputer2 = preprocess_features(solar_wind2, sunspots2 , subset=SOLAR_WIND_FEATURES)

# 确保所有缺失值被填补
assert (features.isna().sum() == 0).all()
assert (features2.isna().sum() == 0).all()

# 目标指标
YCOLS = ["t0", "t1"]

def process_labels(dst):
    y = dst.copy()
    # t0是当前时刻，t1是未来1小时
    y["t0"] = y.groupby("period").dst.shift(0)
    y["t1"] = y.groupby("period").dst.shift(-1)
    return y[YCOLS]

# 生成标签
labels = process_labels(dst)
labels2 = process_labels(dst2)

# 将特征和标签合并
data = labels.join(features)
data2 = labels2.join(features2)

data = pd.concat([data, data2], axis=0)

# 数据集划分
def get_train_test_val(data, test_ratio, val_ratio):
    """将数据划分为训练集、测试集和验证集"""
    # 根据period分组计算
    period_sizes = data.groupby("period").size().reset_index(name="period_size")

    # 初始化DataFrame
    test = pd.DataFrame()
    val = pd.DataFrame()
    train = pd.DataFrame()

    for _, period_data in data.groupby("period"):
        period_size = period_data.shape[0]

        test_size = int(period_size * test_ratio)
        val_size = int(period_size * val_ratio)

        test_data = period_data.iloc[-test_size:]
        val_data = period_data.iloc[-(test_size + val_size):-test_size]
        train_data = period_data.iloc[:-(test_size + val_size)]

        test = pd.concat([test, test_data], axis=0)
        val = pd.concat([val, val_data], axis=0)
        train = pd.concat([train, train_data], axis=0)

    return train, test, val

# 训练集70%，测试集20%，验证集10%
train, test, val = get_train_test_val(data, test_ratio=0.20, val_ratio=0.10)

from torch.utils.data import Dataset, DataLoader

data_config = {
    "timesteps": 128,
    "batch_size": 768,
}


class TimeSeriesDataset(Dataset):
    """自定义时序数据集"""
    def __init__(self, df, timesteps, feature_cols, label_cols):
        """
        Args:
            df: DataFrame格式的数据
            timesteps: 每个序列的时间步步长
            feature_cols: 特征列
            label_cols: 标签列
        """
        self.df = df
        self.timesteps = timesteps
        self.feature_cols = feature_cols
        self.label_cols = label_cols

        # 根据period分组
        self.period_data = []
        for _, period_df in df.groupby("period"):
            # 确保数据量足够
            if len(period_df) <= timesteps:
                continue

            # 时序对齐（特征列与标签列）
            inputs = period_df[feature_cols].values[:-timesteps]
            outputs = period_df[label_cols].values[timesteps:]

            # 创建序列
            sequences = []
            labels = []
            for i in range(len(inputs) - timesteps + 1):
                sequences.append(inputs[i:i+timesteps])
                labels.append(outputs[i])

            if sequences:
                self.period_data.append((np.array(sequences), np.array(labels)))

    def __len__(self):
        return sum(len(seq) for seq, _ in self.period_data)

    def __getitem__(self, idx):
        # 找到索引对应的period和位置
        current_idx = 0
        for sequences, labels in self.period_data:
            if idx < current_idx + len(sequences):
                pos = idx - current_idx
                return (
                    torch.FloatTensor(sequences[pos]),
                    torch.FloatTensor(labels[pos])
                )
            current_idx += len(sequences)
        raise IndexError(f"Index {idx} out of range")


def timeseries_dataset_from_df(df, batch_size, shuffle=False):
    """用DataFrame格式创建DataLoader"""
    timesteps = data_config["timesteps"]

    # 创建数据集
    dataset = TimeSeriesDataset(df, timesteps, XCOLS, YCOLS)

    # 创建 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=0
    )

    return dataloader


# 创建训练和验证DataLoaders
train_ds = timeseries_dataset_from_df(train, data_config["batch_size"])
val_ds = timeseries_dataset_from_df(val, data_config["batch_size"])

print("数据处理完成!\n")

print(f"训练集批次数量: {len(train_ds)}")
print(f"验证集批次数量: {len(val_ds)}")

for batch_x, batch_y in train_ds:
    print(f"特征列形状: {batch_x.shape}")  # (batch_size, timesteps, num_features)
    print(f"标签列形状: {batch_y.shape}")  # (batch_size, num_labels)
    break


# ========== MoE架构定义 ==========

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from tqdm import tqdm

class GateNetwork(nn.Module):
    """CNN+LSTM混合门控网络"""
    def __init__(self, input_dim, timesteps, hidden_dims=[128, 64, 32]):
        super(GateNetwork, self).__init__()

        self.input_dim = input_dim
        self.timesteps = timesteps

        # CNN部分（捕捉局部时序模式）
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.AdaptiveAvgPool1d(1)  # 全局平均池化
        )

        # LSTM部分（捕捉长期依赖）
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # 注意力机制（关注关键时间点）
        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )

        # 特征融合部分
        self.fusion = nn.Sequential(
            nn.Linear(256 + 256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x形状: [batch_size, timesteps, input_dim]
        batch_size = x.size(0)

        # ========== CNN分支 ==========
        x_cnn = x.transpose(1, 2)  # [batch, input_dim, timesteps]
        cnn_features = self.cnn(x_cnn)  # [batch, 256, 1]
        cnn_features = cnn_features.view(batch_size, -1)  # [batch, 256]

        # ========== LSTM分支 ==========
        lstm_out, _ = self.lstm(x)  # [batch, timesteps, 256]

        # 注意力加权
        attention_weights = self.attention(lstm_out)  # [batch, timesteps, 1]
        lstm_attended = lstm_out * attention_weights  # [batch, timesteps, 256]
        lstm_features = torch.sum(lstm_attended, dim=1)  # [batch, 256]

        # ========== 特征融合 ==========
        combined_features = torch.cat([cnn_features, lstm_features], dim=1)  # [batch, 512]
        fused_features = self.fusion(combined_features)  # [batch, 128]

        # ========== 输出层 ==========
        output = self.output_layer(fused_features)  # [batch, 2]

        return output


# 门控数据集
class GateDataset(Dataset):
    """门控网络数据集"""
    def __init__(self, df, timesteps, feature_cols, label_cols, threshold=-50):
        self.timesteps = timesteps
        self.feature_cols = feature_cols
        self.threshold = threshold

        # 从DataFrame中提取序列和标签
        self.sequences, self.gate_labels = self._prepare_data(df, label_cols)

        print(f"数据集: {len(self.sequences)} 个样本")
        print(f"正常样本: {(self.gate_labels == 0).sum()} ({(self.gate_labels == 0).mean():.2%})")
        print(f"异常样本: {(self.gate_labels == 1).sum()} ({(self.gate_labels == 1).mean():.2%})")

    def _prepare_data(self, df, label_cols):
        sequences = []
        gate_labels = []

        for _, period_df in df.groupby("period"):
            if len(period_df) <= self.timesteps:
                continue

            # 提取特征序列
            feature_values = period_df[self.feature_cols].values
            label_values = period_df[label_cols].values

            # 创建序列
            for i in range(len(feature_values) - self.timesteps):
                # 序列特征
                sequence = feature_values[i:i+self.timesteps]
                sequences.append(sequence)

                # 对应的标签（使用序列结束后的第一个标签判断是否异常）
                label_idx = i + self.timesteps
                if label_idx < len(label_values):
                    label = label_values[label_idx]
                    # 如果t0或t1小于阈值，则认为是异常
                    is_abnormal = (label[0] < self.threshold) or (label[1] < self.threshold)
                    gate_labels.append(1 if is_abnormal else 0)
                else:
                    gate_labels.append(0)  # 默认正常

        return np.array(sequences), np.array(gate_labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx])
        label = torch.LongTensor([self.gate_labels[idx]]).squeeze()
        return sequence, label


def prepare_gate_datasets(train_df, val_df, timesteps, feature_cols, label_cols, threshold=-50):
    """准备门控网络数据集"""
    print("准备训练集...")
    train_dataset = GateDataset(train_df, timesteps, feature_cols, label_cols, threshold)

    print("\n准备验证集...")
    val_dataset = GateDataset(val_df, timesteps, feature_cols, label_cols, threshold)

    return train_dataset, val_dataset

# 专家子网
# 正常专家
class NormalExpert(nn.Module):
    def __init__(self, input_dim, timesteps):
        super(NormalExpert, self).__init__()
        self.timesteps = timesteps
        self.input_dim = input_dim

        # ========== 1. TimeDistributed Dense ==========
        self.time_distributed_dense1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=32,
            kernel_size=1,
            padding=0
        )

        # ========== 2. Conv1D层 ==========
        self.conv1 = nn.Conv1d(
            in_channels=32,
            out_channels=32,
            kernel_size=12,
            padding='same'
        )

        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=32,
            kernel_size=12,
            padding='same'
        )

        # ========== 3. 第一个MaxPool1D ==========
        self.maxpool1 = nn.MaxPool1d(
            kernel_size=2,
            stride=2
        )

        # ========== 4. 第二组卷积 ==========
        self.conv3 = nn.Conv1d(
            in_channels=32,
            out_channels=32,
            kernel_size=16,
            padding='same'
        )

        self.conv4 = nn.Conv1d(
            in_channels=32,
            out_channels=32,
            kernel_size=16,
            padding='same'
        )

        # ========== 5. 第二个MaxPool1D ==========
        self.maxpool2 = nn.MaxPool1d(
            kernel_size=2,
            stride=2
        )

        # ========== 6. 第二个TimeDistributed Dense ==========
        self.time_distributed_dense2 = nn.Conv1d(
            in_channels=32,
            out_channels=32,
            kernel_size=1,
            padding=0
        )

        # ========== 7. 计算Flatten后的维度 ==========
        # 经过两次MaxPool1D(2)后，时间步长变为: timesteps // 4
        self.flatten_dim = 32 * (timesteps // 4)

        # ========== 8. Dropout层 ==========
        self.dropout = nn.Dropout(0.3)

        # ========== 9. 全连接层 ==========
        self.fc1 = nn.Linear(self.flatten_dim, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        # x形状: [batch_size, timesteps, input_dim]
        # 转置维度以符合Conv1d的输入格式
        x = x.transpose(1, 2)  # [batch, input_dim, timesteps]

        # 1. TimeDistributed Dense
        x = F.relu(self.time_distributed_dense1(x))  # [batch, 32, timesteps]

        # 2. 第一组Conv1D
        x = F.relu(self.conv1(x))  # [batch, 32, timesteps]
        x = F.relu(self.conv2(x))  # [batch, 32, timesteps]

        # 3. 第一个MaxPool
        x = self.maxpool1(x)  # [batch, 32, timesteps//2]

        # 4. 第二组Conv1D
        x = F.relu(self.conv3(x))  # [batch, 32, timesteps//2]
        x = F.relu(self.conv4(x))  # [batch, 32, timesteps//2]

        # 5. 第二个MaxPool
        x = self.maxpool2(x)  # [batch, 32, timesteps//4]

        # 6. 第二个TimeDistributed Dense
        x = F.relu(self.time_distributed_dense2(x))  # [batch, 32, timesteps//4]

        # 7. Flatten
        x = x.reshape(x.size(0), -1)  # [batch, 32 * (timesteps//4)]

        # 8. Dropout
        x = self.dropout(x)

        # 9. 全连接层
        x = F.relu(self.fc1(x))  # [batch, 256]
        x = self.fc2(x)  # [batch, 2]

        return x

    def get_num_params(self):
        """获取模型参数数量"""
        return sum(p.numel() for p in self.parameters())

# 异常专家
class AbnormalExpert(nn.Module):
    def __init__(self, input_dim, timesteps):
        super(AbnormalExpert, self).__init__()
        self.timesteps = timesteps
        self.input_dim = input_dim
                
        # ========== 1. TimeDistributed Dense ==========
        self.time_distributed_dense1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=32,
            kernel_size=1,
            padding=0
        )
        
        # ========== 2. Conv1D层 ==========
        self.conv1 = nn.Conv1d(
            in_channels=32,
            out_channels=32,
            kernel_size=12,
            padding='same'
        )
        
        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=32,
            kernel_size=12,
            padding='same'
        )
        
        # ========== 3. 第一个MaxPool1D ==========
        self.maxpool1 = nn.MaxPool1d(
            kernel_size=2,
            stride=2
        )
        
        # ========== 4. 第二组卷积 ==========
        self.conv3 = nn.Conv1d(
            in_channels=32,
            out_channels=32,
            kernel_size=16,
            padding='same'
        )
        
        self.conv4 = nn.Conv1d(
            in_channels=32,
            out_channels=32,
            kernel_size=16,
            padding='same'
        )
        
        # ========== 5. 第二个MaxPool1D ==========
        self.maxpool2 = nn.MaxPool1d(
            kernel_size=2,
            stride=2
        )
        
        # ========== 6. 第二个TimeDistributed Dense ==========
        self.time_distributed_dense2 = nn.Conv1d(
            in_channels=32,
            out_channels=32,
            kernel_size=1,
            padding=0
        )
        
        # ========== 7. 计算Flatten后的维度 ==========
        # 经过两次MaxPool1D(2)后，时间步长变为: timesteps // 4
        self.flatten_dim = 32 * (timesteps // 4)
        
        # ========== 8. Dropout层 ==========
        self.dropout = nn.Dropout(0.3)
        
        # ========== 9. 全连接层 ==========
        self.fc1 = nn.Linear(self.flatten_dim, 256)
        self.fc2 = nn.Linear(256, 2)
        
    def forward(self, x):
        # x形状: [batch_size, timesteps, input_dim]
        # 转置维度以符合Conv1d的输入格式
        x = x.transpose(1, 2)  # [batch, input_dim, timesteps]
        
        # 1. TimeDistributed Dense
        x = F.relu(self.time_distributed_dense1(x))  # [batch, 32, timesteps]
        
        # 2. 第一组Conv1D
        x = F.relu(self.conv1(x))  # [batch, 32, timesteps]
        x = F.relu(self.conv2(x))  # [batch, 32, timesteps]
        
        # 3. 第一个MaxPool
        x = self.maxpool1(x)  # [batch, 32, timesteps//2]
        
        # 4. 第二组Conv1D
        x = F.relu(self.conv3(x))  # [batch, 32, timesteps//2]
        x = F.relu(self.conv4(x))  # [batch, 32, timesteps//2]
        
        # 5. 第二个MaxPool
        x = self.maxpool2(x)  # [batch, 32, timesteps//4]
        
        # 6. 第二个TimeDistributed Dense
        x = F.relu(self.time_distributed_dense2(x))  # [batch, 32, timesteps//4]
        
        # 7. Flatten
        x = x.reshape(x.size(0), -1)  # [batch, 32 * (timesteps//4)]
        
        # 8. Dropout
        x = self.dropout(x)
        
        # 9. 全连接层
        x = F.relu(self.fc1(x))  # [batch, 256]
        x = self.fc2(x)  # [batch, 2]
        
        return x
    
    def get_num_params(self):
        """获取模型参数数量"""
        return sum(p.numel() for p in self.parameters())


# ========== MoE测试 ==========

from tqdm import tqdm
import json

print("="*70)
print("完整MoE系统测试")
print("="*70)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

class MoESystem(nn.Module):
    """完整的MoE系统"""
    def __init__(self, gate_model, normal_expert, abnormal_expert, confidence_threshold=0.7):
        super(MoESystem, self).__init__()
        self.gate_model = gate_model
        self.normal_expert = normal_expert
        self.abnormal_expert = abnormal_expert
        self.confidence_threshold = confidence_threshold

    def forward(self, x, mode='adaptive'):
        """
        Args:
            x: 输入序列 [batch_size, timesteps, input_dim]
            mode: 'gate' (直接使用门控) | 'adaptive' (自适应门控) | 'perfect' (完美门控)
            y_true: 真实标签（仅完美门控模式需要）
        """
        batch_size = x.size(0)

        if mode == 'perfect':
            # 完美门控模式需要额外的y_true参数
            return None  # 这个模式在外部处理

        # 获取门控决策
        gate_outputs = self.gate_model(x)
        gate_probs, gate_decisions = torch.max(gate_outputs, dim=1)

        outputs = torch.zeros(batch_size, 2, device=x.device)

        if mode == 'gate':
            # 直接门控模式：完全相信门控网络
            normal_mask = gate_decisions == 0
            abnormal_mask = gate_decisions == 1

        elif mode == 'adaptive':
            # 自适应门控模式：高置信度用门控，低置信度用正常专家（保守）
            high_confidence_mask = gate_probs >= self.confidence_threshold
            low_confidence_mask = gate_probs < self.confidence_threshold

            # 高置信度样本使用门控决策
            high_conf_normal = high_confidence_mask & (gate_decisions == 0)
            high_conf_abnormal = high_confidence_mask & (gate_decisions == 1)

            # 低置信度样本使用正常专家（保守策略）
            low_conf_all = low_confidence_mask

            normal_mask = high_conf_normal | low_conf_all
            abnormal_mask = high_conf_abnormal

        # 使用对应专家进行预测
        if normal_mask.any():
            normal_x = x[normal_mask]
            outputs[normal_mask] = self.normal_expert(normal_x)

        if abnormal_mask.any():
            abnormal_x = x[abnormal_mask]
            outputs[abnormal_mask] = self.abnormal_expert(abnormal_x)

        return outputs, gate_decisions, gate_probs


def load_all_models():
    """加载所有模型"""
    print("加载模型...")

    input_dim = len(XCOLS)
    timesteps = data_config["timesteps"]

    # 加载门控网络
    gate_model = GateNetwork(input_dim, timesteps).to(device)
    gate_checkpoint = torch.load('gate_checkpoints/best_gate_model.pth', map_location=device)
    gate_model.load_state_dict(gate_checkpoint['model_state_dict'])
    gate_model.eval()
    print(f"门控网络加载成功 (准确率: {gate_checkpoint.get('val_accuracy', 'N/A'):.4f})")

    # 加载正常专家
    normal_expert = NormalExpert(input_dim, timesteps).to(device)
    normal_checkpoint = torch.load('expert_checkpoints/normal_expert_best.pth', map_location=device)
    normal_expert.load_state_dict(normal_checkpoint['model_state_dict'])
    normal_expert.eval()
    print(f"正常专家加载成功 (验证损失: {normal_checkpoint.get('best_value', 'N/A'):.6f})")

    # 加载异常专家
    abnormal_expert = AbnormalExpert(input_dim, timesteps).to(device)
    abnormal_checkpoint = torch.load('expert_checkpoints/abnormal_expert_best.pth', map_location=device)
    abnormal_expert.load_state_dict(abnormal_checkpoint['model_state_dict'])
    abnormal_expert.eval()
    print(f"异常专家加载成功 (验证损失: {abnormal_checkpoint.get('best_value', 'N/A'):.6f})")

    return gate_model, normal_expert, abnormal_expert


def prepare_test_data(test_df):
    """准备测试数据"""
    print("\n准备测试数据...")

    test_sequences = []
    test_labels = []

    for _, period_df in test_df.groupby("period"):
        if len(period_df) <= data_config["timesteps"]:
            continue

        inputs = period_df[XCOLS].values[:-data_config["timesteps"]]
        outputs = period_df[YCOLS].values[data_config["timesteps"]:]

        for i in range(len(inputs) - data_config["timesteps"] + 1):
            test_sequences.append(inputs[i:i+data_config["timesteps"]])
            test_labels.append(outputs[i])

    test_sequences = np.array(test_sequences)
    test_labels = np.array(test_labels)

    print(f"测试集: {len(test_sequences)} 个样本")

    # 标记异常样本
    is_abnormal = (test_labels < -50).any(axis=1)
    print(f"正常样本: {(~is_abnormal).sum()} ({(~is_abnormal).mean()*100:.2f}%)")
    print(f"异常样本: {is_abnormal.sum()} ({is_abnormal.mean()*100:.2f}%)")

    return test_sequences, test_labels, is_abnormal


def test_moe_system(moe_system, test_sequences, test_labels, mode='adaptive'):
    """测试MoE系统"""
    print(f"\n测试MoE系统 (模式: {mode})...")

    # 创建DataLoader
    dataset = TensorDataset(
        torch.FloatTensor(test_sequences),
        torch.FloatTensor(test_labels)
    )
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

    criterion = nn.MSELoss()

    all_predictions = []
    all_labels = []
    all_gate_decisions = []
    all_gate_confidences = []

    with torch.no_grad():
        for batch_x, batch_y in tqdm(dataloader, desc="测试"):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            if mode == 'perfect':
                # 完美门控：根据真实标签选择专家
                is_abnormal_batch = (batch_y < -50).any(dim=1)

                outputs = torch.zeros_like(batch_y, device=device)

                normal_mask = ~is_abnormal_batch
                abnormal_mask = is_abnormal_batch

                if normal_mask.any():
                    normal_outputs = moe_system.normal_expert(batch_x[normal_mask])
                    outputs[normal_mask] = normal_outputs

                if abnormal_mask.any():
                    abnormal_outputs = moe_system.abnormal_expert(batch_x[abnormal_mask])
                    outputs[abnormal_mask] = abnormal_outputs

                gate_decisions = is_abnormal_batch.long()
                gate_confidences = torch.ones_like(is_abnormal_batch.float())

            else:
                # 门控或自适应门控
                outputs, gate_decisions, gate_confidences = moe_system(batch_x, mode=mode)

            all_predictions.append(outputs.cpu())
            all_labels.append(batch_y.cpu())
            all_gate_decisions.append(gate_decisions.cpu())
            all_gate_confidences.append(gate_confidences.cpu())

    # 合并结果
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_gate_decisions = torch.cat(all_gate_decisions, dim=0)
    all_gate_confidences = torch.cat(all_gate_confidences, dim=0)

    # 计算指标
    mse = criterion(all_predictions, all_labels).item()
    rmse = np.sqrt(mse)

    # 计算门控准确率（完美门控模式除外）
    if mode != 'perfect':
        true_abnormal = (all_labels < -50).any(dim=1)
        gate_accuracy = (all_gate_decisions == true_abnormal.long()).float().mean().item()
    else:
        gate_accuracy = 1.0  # 完美门控准确率为1

    # 分别计算正常和异常样本的性能
    true_abnormal = (all_labels < -50).any(dim=1)

    if true_abnormal.any():
        abnormal_predictions = all_predictions[true_abnormal]
        abnormal_labels = all_labels[true_abnormal]
        abnormal_mse = criterion(abnormal_predictions, abnormal_labels).item()
        abnormal_rmse = np.sqrt(abnormal_mse)
    else:
        abnormal_rmse = 0

    normal_predictions = all_predictions[~true_abnormal]
    normal_labels = all_labels[~true_abnormal]

    if normal_predictions.numel() > 0:
        normal_mse = criterion(normal_predictions, normal_labels).item()
        normal_rmse = np.sqrt(normal_mse)
    else:
        normal_rmse = 0

    return {
        'mse': mse,
        'rmse': rmse,
        'normal_rmse': normal_rmse,
        'abnormal_rmse': abnormal_rmse,
        'gate_accuracy': gate_accuracy,
        'gate_decisions': all_gate_decisions,
        'gate_confidences': all_gate_confidences,
        'predictions': all_predictions,
        'labels': all_labels
    }


def plot_moe_results(all_results):
    """绘制MoE系统结果"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    modes = list(all_results.keys())
    rmse_values = [results['rmse'] for results in all_results.values()]

    # 不同模式的RMSE对比
    axes[0, 0].bar(modes, rmse_values)
    axes[0, 0].set_xlabel('Gate Mode')
    axes[0, 0].set_ylabel('RMSE')
    axes[0, 0].set_title('RMSE Comparison of Different Gate Modes')
    axes[0, 0].grid(True, alpha=0.3)

    # 添加数值标签
    for i, v in enumerate(rmse_values):
        axes[0, 0].text(i, v, f'{v:.4f}', ha='center', va='bottom')

    # 门控决策分布
    if 'gate' in all_results:
        gate_decisions = all_results['gate']['gate_decisions']
        normal_count = (gate_decisions == 0).sum().item()
        abnormal_count = (gate_decisions == 1).sum().item()

        axes[0, 1].pie([normal_count, abnormal_count],
                      labels=['NormalExpert', 'AbnormalExpert'],
                      autopct='%1.1f%%')
        axes[0, 1].set_title('Gate Decision Distribution')

    # 预测vs真实值散点图
    if 'adaptive' in all_results:
        predictions = all_results['adaptive']['predictions']
        labels = all_results['adaptive']['labels']

        axes[0, 2].scatter(labels[:, 0].numpy(), predictions[:, 0].numpy(),
                          alpha=0.5, s=2)
        axes[0, 2].plot([-100, 50], [-100, 50], 'r--', alpha=0.5)
        axes[0, 2].set_xlabel('True Value t0')
        axes[0, 2].set_ylabel('Predicted Value t0')
        axes[0, 2].set_title('t0: Prediction vs True')
        axes[0, 2].grid(True, alpha=0.3)

    # 残差分布
    if 'adaptive' in all_results:
        residuals_t0 = predictions[:, 0] - labels[:, 0]
        residuals_t1 = predictions[:, 1] - labels[:, 1]

        axes[1, 0].hist(residuals_t0.numpy(), bins=100, alpha=0.7, label='t0 Residual')
        axes[1, 0].hist(residuals_t1.numpy(), bins=100, alpha=0.7, label='t1 Residual')
        axes[1, 0].set_xlabel('Residual Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Prediction Residual Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # 门控置信度分布
    if 'gate' in all_results:
        confidences = all_results['gate']['gate_confidences']
        axes[1, 1].hist(confidences.numpy(), bins=50)
        axes[1, 1].set_xlabel('Gate Confidence')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Gate Confidence Distribution')
        axes[1, 1].grid(True, alpha=0.3)

    # 与基线对比
    baseline_rmse = 10.091585
    axes[1, 2].bar(['Single Model', 'MoE System'], [baseline_rmse, rmse_values[0]])
    axes[1, 2].set_ylabel('RMSE')
    axes[1, 2].set_title('MoE System vs Single Model')
    axes[1, 2].grid(True, alpha=0.3)

    # 添加数值标签
    axes[1, 2].text(0, baseline_rmse, f'{baseline_rmse:.4f}', ha='center', va='bottom')
    axes[1, 2].text(1, rmse_values[0], f'{rmse_values[0]:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('moe_system_results.png', dpi=150, bbox_inches='tight')
    


def test_different_confidence_thresholds(gate_model, normal_expert, abnormal_expert, test_sequences, test_labels):
    """测试不同置信度阈值"""
    print("\n" + "="*70)
    print("测试不同置信度阈值")
    print("="*70)

    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    results = {}

    for threshold in thresholds:
        print(f"\n测试置信度阈值: {threshold}")
        moe_system = MoESystem(gate_model, normal_expert, abnormal_expert, confidence_threshold=threshold)
        results[threshold] = test_moe_system(moe_system, test_sequences, test_labels, mode='adaptive')

        print(f"  RMSE: {results[threshold]['rmse']:.6f}")
        print(f"  门控准确率: {results[threshold]['gate_accuracy']:.4f}")

    # 找到最佳阈值
    best_threshold = min(results.keys(), key=lambda t: results[t]['rmse'])
    best_rmse = results[best_threshold]['rmse']

    print(f"\n最佳置信度阈值: {best_threshold}")
    print(f"最佳RMSE: {best_rmse:.6f}")

    # 绘制阈值 vs RMSE曲线
    thresholds_list = list(results.keys())
    rmse_values = [results[t]['rmse'] for t in thresholds_list]

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds_list, rmse_values, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Confidence Threshold')
    plt.ylabel('RMSE')
    plt.title('MoE System Performance under Different Confidence Thresholds')
    plt.grid(True, alpha=0.3)

    # 标记最佳点
    best_idx = thresholds_list.index(best_threshold)
    plt.scatter(best_threshold, best_rmse, color='red', s=100,
                zorder=5, label=f'Best Point ({best_threshold}, {best_rmse:.4f})')

    plt.legend()
    plt.tight_layout()
    plt.savefig('confidence_threshold_vs_rmse.png', dpi=150, bbox_inches='tight')
    

    return results, best_threshold


def save_final_results(all_results, baseline_rmse=10.091585):
    """保存最终结果"""
    print("\n" + "="*70)
    print("保存最终结果")
    print("="*70)

    final_results = {
        'baseline': {
            'rmse': baseline_rmse,
            'description': '单模型Conv1DTimeDistributedNet'
        },
        'moe_systems': {}
    }

    for mode, results in all_results.items():
        improvement = (baseline_rmse - results['rmse']) / baseline_rmse * 100

        final_results['moe_systems'][mode] = {
            'rmse': float(results['rmse']),
            'normal_rmse': float(results.get('normal_rmse', 0)),
            'abnormal_rmse': float(results.get('abnormal_rmse', 0)),
            'gate_accuracy': float(results.get('gate_accuracy', 0)),
            'improvement_percent': float(improvement),
            'description': {
                'gate': '直接门控',
                'adaptive': '自适应门控',
                'perfect': '完美门控'
            }.get(mode, mode)
        }

    # 保存到JSON文件
    with open('moe_final_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)

    # 打印总结
    print("\n最终结果总结:")
    print(f"{'模式':<15} {'RMSE':<10} {'改进%':<10} {'门控准确率':<12}")
    print("-" * 50)

    for mode, info in final_results['moe_systems'].items():
        print(f"{mode:<15} {info['rmse']:<10.6f} {info['improvement_percent']:<10.2f} {info['gate_accuracy']:<12.4f}")



# 主测试函数
def main():
    """主测试函数"""

    # 加载所有模型
    gate_model, normal_expert, abnormal_expert = load_all_models()

    # 准备测试数据
    test_sequences, test_labels, is_abnormal = prepare_test_data(test)

    # 测试不同门控模式
    print("\n" + "="*70)
    print("测试不同门控模式")
    print("="*70)

    all_results = {}

    # 创建MoE系统
    moe_system = MoESystem(gate_model, normal_expert, abnormal_expert, confidence_threshold=0.7)

    # 测试完美门控
    print("\n1. 完美门控模式 (理论上限)...")
    perfect_results = test_moe_system(moe_system, test_sequences, test_labels, mode='perfect')
    all_results['perfect'] = perfect_results
    print(f"  完美门控RMSE: {perfect_results['rmse']:.6f}")

    # 测试直接门控
    print("\n2. 直接门控模式...")
    gate_results = test_moe_system(moe_system, test_sequences, test_labels, mode='gate')
    all_results['gate'] = gate_results
    print(f"  直接门控RMSE: {gate_results['rmse']:.6f}")
    print(f"  门控准确率: {gate_results['gate_accuracy']:.4f}")

    # 测试自适应门控
    print("\n3. 自适应门控模式 (置信度阈值=0.7)...")
    adaptive_results = test_moe_system(moe_system, test_sequences, test_labels, mode='adaptive')
    all_results['adaptive'] = adaptive_results
    print(f"  自适应门控RMSE: {adaptive_results['rmse']:.6f}")
    print(f"  门控准确率: {adaptive_results['gate_accuracy']:.4f}")

    # 测试不同置信度阈值
    threshold_results, best_threshold = test_different_confidence_thresholds(
        gate_model, normal_expert, abnormal_expert, test_sequences, test_labels
    )

    # 使用最佳阈值重新测试
    print(f"\n使用最佳阈值 ({best_threshold}) 重新测试...")
    best_moe_system = MoESystem(gate_model, normal_expert, abnormal_expert, confidence_threshold=best_threshold)
    best_adaptive_results = test_moe_system(best_moe_system, test_sequences, test_labels, mode='adaptive')
    all_results[f'adaptive_best({best_threshold})'] = best_adaptive_results


    # 绘制结果
    plot_moe_results(all_results)

    # 保存最终结果
    save_final_results(all_results, baseline_rmse=10.091585)


# 运行测试
if __name__ == "__main__":
    all_test_results = main()
