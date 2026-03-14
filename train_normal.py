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


# ========== 模型架构 ==========

import torch.nn as nn
import torch.nn.functional as F

# 虚拟模型
class DummyModel(nn.Module):
    """虚拟模型，参数比真实模型稍大，仅用于初始化，不用于训练"""
    def __init__(self, input_dim, timesteps):
        super(DummyModel, self).__init__()
        self.timesteps = timesteps
        self.input_dim = input_dim

        # ========== 1. TimeDistributed Dense ==========
        self.time_distributed_dense1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=48,
            kernel_size=1,
            padding=0
        )

        # ========== 2. Conv1D层 ==========
        self.conv1 = nn.Conv1d(
            in_channels=48,
            out_channels=48,
            kernel_size=12,
            padding='same'
        )

        self.conv2 = nn.Conv1d(
            in_channels=48,
            out_channels=48,
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
            in_channels=48,
            out_channels=48,
            kernel_size=16,
            padding='same'
        )

        self.conv4 = nn.Conv1d(
            in_channels=48,
            out_channels=48,
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
            in_channels=48,
            out_channels=48,
            kernel_size=1,
            padding=0
        )

        # ========== 7. 计算Flatten后的维度 ==========
        # 经过两次MaxPool1D(2)后，时间步长变为: timesteps // 4
        self.flatten_dim = 48 * (timesteps // 4) 

        # ========== 8. Dropout层 ==========
        self.dropout = nn.Dropout(0.3)

        # ========== 9. 全连接层 ==========
        self.fc1 = nn.Linear(self.flatten_dim, 384)
        self.fc2 = nn.Linear(384, 2)

    def forward(self, x):
        # x形状: [batch_size, timesteps, input_dim]
        x = x.transpose(1, 2)  # [batch, input_dim, timesteps]

        # 1. TimeDistributed Dense
        x = F.relu(self.time_distributed_dense1(x))  # [batch, 48, timesteps]

        # 2. 第一组Conv1D
        x = F.relu(self.conv1(x))  # [batch, 48, timesteps]
        x = F.relu(self.conv2(x))  # [batch, 48, timesteps]

        # 3. 第一个MaxPool
        x = self.maxpool1(x)  # [batch, 48, timesteps//2]

        # 4. 第二组Conv1D
        x = F.relu(self.conv3(x))  # [batch, 48, timesteps//2]
        x = F.relu(self.conv4(x))  # [batch, 48, timesteps//2]

        # 5. 第二个MaxPool
        x = self.maxpool2(x)  # [batch, 48, timesteps//4]

        # 6. 第二个TimeDistributed Dense
        x = F.relu(self.time_distributed_dense2(x))  # [batch, 48, timesteps//4]

        # 7. Flatten
        x = x.reshape(x.size(0), -1)  # [batch, 48 * (timesteps//4)]

        # 8. Dropout
        x = self.dropout(x)

        # 9. 全连接层
        x = F.relu(self.fc1(x))  # [batch, 384]
        x = self.fc2(x)  # [batch, 2]

        return x

def initialize_actual_model_with_dummy(actual_model, dummy_model):
    """用虚拟模型的参数初始化实际模型（截断方式：当形状不匹配时，用虚拟模型的前若干维填充）"""
    actual_state_dict = actual_model.state_dict()
    dummy_state_dict = dummy_model.state_dict()

    for name, param in dummy_state_dict.items():
        if name in actual_state_dict:
            actual_param = actual_state_dict[name]
            dummy_param = param

            # 如果形状完全相同，直接复制
            if actual_param.shape == dummy_param.shape:
                actual_state_dict[name] = dummy_param.clone()
            else:
                # 形状不匹配时，截断复制（只复制公共部分）
                # 根据维度数分别处理
                if len(actual_param.shape) == 1:  # 1D参数（如偏置）
                    min_dim = min(actual_param.shape[0], dummy_param.shape[0])
                    actual_param.data[:min_dim].copy_(dummy_param[:min_dim])
                
                elif len(actual_param.shape) == 2:  # 2D参数（全连接层权重）
                    min_out = min(actual_param.shape[0], dummy_param.shape[0])
                    min_in = min(actual_param.shape[1], dummy_param.shape[1])
                    actual_param.data[:min_out, :min_in].copy_(
                        dummy_param[:min_out, :min_in]
                    )
                
                elif len(actual_param.shape) == 3:  # 3D参数（卷积层权重）
                    min_out = min(actual_param.shape[0], dummy_param.shape[0])
                    min_in = min(actual_param.shape[1], dummy_param.shape[1])
                    min_kernel = min(actual_param.shape[2], dummy_param.shape[2])
                    actual_param.data[:min_out, :min_in, :min_kernel].copy_(
                        dummy_param[:min_out, :min_in, :min_kernel]
                    )
                    
    actual_model.load_state_dict(actual_state_dict)
    return actual_model

class ActualModel(nn.Module):
    def __init__(self, input_dim, timesteps):
        super(ActualModel, self).__init__()
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
        self.fc2 = nn.Linear(256, 6)

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
        x = self.fc2(x)  # [batch, 6]

        return x.view(-1,2,3)

input_dim = len(XCOLS)
timesteps = data_config["timesteps"]

# 创建模型
actual_model = ActualModel(input_dim, timesteps)
dummy_model = DummyModel(input_dim, timesteps)

print("\n模型加载中...")
print(f"模型参数数量（实际模型）: {sum(p.numel() for p in actual_model.parameters())}")


# ========== 主训练与模型评估 ==========

# 具有物理意义的损失函数
class QuantileLoss(nn.Module):
    """分位数损失函数"""
    def __init__(self, quantiles=[0.05, 0.5, 0.95]):
        super(QuantileLoss, self).__init__()
        self.quantiles = quantiles

    def forward(self, predictions, targets):
        """
        predictions: [batch, 2, n_quantiles]
        targets: [batch, 2]
        """
        loss = 0.0
        for i, q in enumerate(self.quantiles):
            pred_q = predictions[:, :, i]
            error = targets - pred_q
            loss_q = torch.max(q * error, (q - 1) * error)
            loss += loss_q.mean()
        return loss

# 自定义学习率调度器
class ReduceLRBacktrack:
    """带回溯功能的学习率调度器"""
    def __init__(self, model, best_model_path, optimizer,
                 monitor='val_loss', factor=0.5, patience=5,
                 min_lr=1e-10, mode='min', verbose=True):
        """
            model: 要训练的模型
            best_model_path: 最佳模型保存路径
            optimizer: 优化器
            monitor: 监控指标
            factor: 学习率衰减因子
            patience: 容忍多少个epoch没有改善
            min_lr: 最小学习率
            mode: 'min' 表示指标越小越好，'max' 表示指标越大越好
            verbose: 是否打印信息
        """
        self.model = model
        self.best_model_path = best_model_path
        self.optimizer = optimizer
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.mode = mode
        self.verbose = verbose

        # 初始化内部状态
        self.wait = 0
        self.best = float('inf') if mode == 'min' else -float('inf')
        self.best_epoch = 0
        self.current_lr = optimizer.param_groups[0]['lr']

        # 确保目录存在
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    def step(self, current_value, epoch):
        """
            current_value: 当前监控指标的值
            epoch: 当前epoch数
        """
        # 检查是否改善
        if self.mode == 'min':
            is_better = current_value < self.best
        else:  # mode == 'max'
            is_better = current_value > self.best

        if is_better:
            self.best = current_value
            self.best_epoch = epoch
            self.wait = 0

            # 保存当前最佳模型
            self._save_best_model()

            if self.verbose:
                print(f"Epoch {epoch}: 监控指标改善 -> {current_value:.6f}")

        else:
            # 没有改善，等待计数加1
            self.wait += 1

            if self.verbose:
                print(f"Epoch {epoch}: 监控指标未改善 ({self.wait}/{self.patience})")

            # 检查是否需要降低学习率
            if self.wait >= self.patience:
                if self.verbose:
                    print(f"Epoch {epoch}: 容忍 {self.patience} 个epoch无改善，触发回溯")

                # 加载最佳模型
                self._load_best_model()

                # 降低学习率
                old_lr = self.current_lr
                new_lr = max(old_lr * self.factor, self.min_lr)

                if new_lr < old_lr:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_lr

                    self.current_lr = new_lr
                    self.wait = 0  # 重置等待计数

                    if self.verbose:
                        print(f"  学习率从 {old_lr:.2e} 降低到 {new_lr:.2e}")
                        print(f"  回溯到第 {self.best_epoch} 个epoch的最佳模型")
                else:
                    if self.verbose:
                        print(f"  学习率已达到最小值 {self.min_lr:.2e}")

    def _save_best_model(self):
        """保存最佳模型"""
        torch.save({
            'epoch': self.best_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_value': self.best,
            'current_lr': self.current_lr,
        }, self.best_model_path)

    def _load_best_model(self):
        """加载最佳模型"""
        if os.path.exists(self.best_model_path):
            checkpoint = torch.load(self.best_model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if self.verbose:
                print(f"  已加载第 {checkpoint['epoch']} 个epoch的最佳模型")
        else:
            if self.verbose:
                print(f"  警告: 找不到最佳模型文件 {self.best_model_path}")

    def get_last_lr(self):
        """获取当前学习率"""
        return [self.current_lr]


import torch.optim as optim
from tqdm import tqdm
import os
import time

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 模型配置
model_config = {"n_epochs": 30}

# 用虚拟模型进行模型初始化
actual_model = initialize_actual_model_with_dummy(actual_model, dummy_model)
model = actual_model.to(device)
print("模型初始化完成！\n")

# 损失函数和优化器
class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        return torch.sqrt(torch.mean((y_pred - y_true)**2))

criterion = QuantileLoss(quantiles=[0.05, 0.5, 0.95]).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 学习率调度器
scheduler = ReduceLRBacktrack(
        model=model,
        best_model_path='expert_checkpoints/normal_expert_best.pth',
        optimizer=optimizer,
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-10,
        mode='min',
        verbose=True
    )

# 训练和验证函数
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    batch_count = 0

    progress_bar = tqdm(dataloader, desc="训练", leave=False)
    for batch_x, batch_y in progress_bar:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)

        loss = criterion(outputs, batch_y)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        batch_count += 1
        progress_bar.set_postfix(loss=loss.item())

    return running_loss / max(batch_count, 1)

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    batch_count = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="验证", leave=False)
        for batch_x, batch_y in progress_bar:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            running_loss += loss.item()
            batch_count += 1
            progress_bar.set_postfix(loss=loss.item())

    return running_loss / max(batch_count, 1)

# 训练主循环
best_val_loss = float('inf')
history = {'train_loss': [], 'val_loss': [], 'learning_rate': []}

print("开始训练...")
for epoch in range(model_config["n_epochs"]):
    print(f"\nEpoch {epoch+1}/{model_config['n_epochs']}")
    start_time = time.time()

    # 训练
    train_loss = train_epoch(model, train_ds, criterion, optimizer, device)

    # 验证
    val_loss = validate_epoch(model, val_ds, criterion, device)

    scheduler.step(val_loss, epoch+1)

    if val_loss < best_val_loss:
        best_val_loss = val_loss

    # 记录历史
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['learning_rate'].append(optimizer.param_groups[0]['lr'])

    # 打印epoch结果
    epoch_time = time.time() - start_time
    print(f"训练损失: {train_loss:.6f}, 验证损失: {val_loss:.6f}, "
          f"学习率: {optimizer.param_groups[0]['lr']:.2e}, 时间: {epoch_time:.1f}s")

    # 提前停止检查
    if optimizer.param_groups[0]['lr'] <= 1e-10:
        print("学习率已降至最小值，停止训练")
        break

print(f"\n训练完成，最佳验证损失: {best_val_loss:.6f}")

# 设置matplotlib的backend为Agg
plt.switch_backend('Agg')

# 绘制训练曲线
plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('NormalExpert Training Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(np.sqrt(history['train_loss']), label='Train RMSE')
plt.plot(np.sqrt(history['val_loss']), label='Val RMSE')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.title('NormalExpert RMSE')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.plot(history['learning_rate'])
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.yscale('log')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('normal_expert_training.png', dpi=150, bbox_inches='tight')

# ========== 模型测试 ==========

print("\n开始模型测试...")

# 加载最佳模型
def load_best_model(model_path='expert_checkpoints/normal_expert_best.pth'):
    checkpoint = torch.load(model_path, map_location=device)
    model = ActualModel(input_dim=len(XCOLS), timesteps=data_config["timesteps"]).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"加载最佳模型 (epoch {checkpoint['epoch']}, val_loss={checkpoint['best_value']:.6f})")
    return model

model = load_best_model()

# 创建测试集DataLoader
test_ds = timeseries_dataset_from_df(test, data_config["batch_size"])
print(f"测试集批次数量: {len(test_ds)}")

# 在测试集上评估
def evaluate_simple(model, test_dataloader, device):
    model.eval()
    running_loss = 0.0
    batch_count = 0

    with torch.no_grad():
        for batch_x, batch_y in test_dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x) # 形状 [batch, 2, 3]
            
            # 使用中位数预测作为最终点估计计算 MSE
            median_output = outputs[:, :, 1] 
            loss = F.mse_loss(median_output, batch_y)

            running_loss += loss.item()
            batch_count += 1
    return running_loss / max(batch_count, 1)

test_mse = evaluate_simple(model, test_ds, device)
test_rmse = np.sqrt(test_mse)

print(f"测试集MSE: {test_mse:.6f}")
print(f"测试集RMSE: {test_rmse:.6f}")
