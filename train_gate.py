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
torch.manual_seed(42)
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


# ========== 门控架构 ==========

import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

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

# ========== 主训练 ==========

print("="*70)
print("训练门控网络")
print("="*70)

# 准备数据集
train_dataset, val_dataset = prepare_gate_datasets(
    train, val,
    timesteps=data_config["timesteps"],
    feature_cols=XCOLS,
    label_cols=YCOLS,
    threshold=-50
)

# 创建DataLoader
batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 创建模型
input_dim = len(XCOLS)
timesteps = data_config["timesteps"]
model = GateNetwork(input_dim, timesteps).to(device)

# 损失函数和优化器
import torch.nn.functional as F

# 自定义FocalLoss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: 类别权重系数
            gamma: 调节因子
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 计算交叉熵
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 获取预测概率 pt
        pt = torch.exp(-ce_loss) 
        
        # 计算 Focal Loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

criterion = FocalLoss(alpha=0.8, gamma=2.0)
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)

# 学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3
)

# 训练参数
n_epochs = 50
best_val_accuracy = 0.0
patience = 10
patience_counter = 0

# 创建保存目录
os.makedirs('gate_checkpoints', exist_ok=True)

# 训练历史
history = {
    'train_loss': [], 'val_loss': [],
    'train_acc': [], 'val_acc': [],
    'learning_rate': []
}

print(f"\n开始训练，共 {n_epochs} 个epoch")
print(f"训练集大小: {len(train_dataset)}，批次大小: {batch_size}")
print(f"验证集大小: {len(val_dataset)}")

for epoch in range(n_epochs):
    print(f"\nEpoch {epoch+1}/{n_epochs}")

    # 训练阶段
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    progress_bar = tqdm(train_loader, desc="训练", leave=False)
    for sequences, labels in progress_bar:
        sequences, labels = sequences.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 统计
        train_loss += loss.item() * sequences.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

        # 更新进度条
        progress_bar.set_postfix({
            'loss': loss.item(),
            'acc': (predicted == labels).float().mean().item()
        })

    avg_train_loss = train_loss / len(train_dataset)
    train_accuracy = train_correct / train_total

    # 验证阶段
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="验证", leave=False)
        for sequences, labels in progress_bar:
            sequences, labels = sequences.to(device), labels.to(device)

            outputs = model(sequences)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * sequences.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': (predicted == labels).float().mean().item()
            })

    avg_val_loss = val_loss / len(val_dataset)
    val_accuracy = val_correct / val_total

    # 学习率调整
    scheduler.step(val_accuracy)
    current_lr = optimizer.param_groups[0]['lr']

    # 记录历史
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['train_acc'].append(train_accuracy)
    history['val_acc'].append(val_accuracy)
    history['learning_rate'].append(current_lr)

    # 打印epoch结果
    print(f"训练损失: {avg_train_loss:.4f}, 训练准确率: {train_accuracy:.4f}")
    print(f"验证损失: {avg_val_loss:.4f}, 验证准确率: {val_accuracy:.4f}")
    print(f"学习率: {current_lr:.6f}")

    # 保存最佳模型
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        patience_counter = 0

        # 保存完整模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_accuracy': val_accuracy,
            'train_accuracy': train_accuracy,
            'history': history,
        }, 'gate_checkpoints/best_gate_model.pth')

        print(f"保存最佳模型，验证准确率: {val_accuracy:.4f}")
    else:
        patience_counter += 1
        print(f"验证准确率未提升，耐心计数: {patience_counter}/{patience}")

    # 定期保存检查点
    if (epoch + 1) % 5 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_accuracy': val_accuracy,
            'train_accuracy': train_accuracy,
        }, f'gate_checkpoints/gate_model_epoch_{epoch+1}.pth')

    # 早停检查
    if patience_counter >= patience:
        print(f"验证准确率连续 {patience} 个epoch未提升，停止训练")
        break

    # 如果学习率过低，停止训练
    if current_lr < 1e-7:
        print("学习率过低，停止训练")
        break

print(f"\n训练完成，最佳验证准确率: {best_val_accuracy:.4f}")

def plot_training_history(history):
    """绘制训练历史"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 损失曲线
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 准确率曲线
    axes[1].plot(history['train_acc'], label='Train Accuracy')
    axes[1].plot(history['val_acc'], label='Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 学习率曲线
    axes[2].plot(history['learning_rate'])
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('gate_training_history.png', dpi=150, bbox_inches='tight')
    

# 绘制训练曲线
plot_training_history(history)


# ========== 门控测试 ==========

def load_gate_model(model_path='gate_checkpoints/best_gate_model.pth'):
    """加载门控网络"""
    print("="*70)
    print("加载并测试门控网络")
    print("="*70)

    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}")
        print("请先训练门控网络或提供正确的模型路径")
        return None

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载模型
    checkpoint = torch.load(model_path, map_location=device)

    # 创建模型
    input_dim = len(XCOLS)
    timesteps = data_config["timesteps"]
    model = GateNetwork(input_dim, timesteps).to(device)

    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"模型加载成功")
    print(f"训练epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"验证准确率: {checkpoint.get('val_accuracy', 'N/A'):.4f}")
    print(f"训练准确率: {checkpoint.get('train_accuracy', 'N/A'):.4f}")

    return model, checkpoint


def test_gate_performance(model, test_df):
    """测试门控网络性能"""
    print("\n准备测试数据...")

    # 创建测试数据集
    test_dataset = GateDataset(
        test_df,
        timesteps=data_config["timesteps"],
        feature_cols=XCOLS,
        label_cols=YCOLS,
        threshold=-50
    )

    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # 测试模型
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []

    print("进行预测...")
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)

            outputs = model(sequences)
            probabilities = outputs
            _, predictions = torch.max(outputs, 1)

            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
            all_probabilities.append(probabilities.cpu())

    # 合并结果
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_probabilities = torch.cat(all_probabilities, dim=0)

    # 计算准确率
    accuracy = (all_predictions == all_labels).float().mean().item()

    # 计算混淆矩阵
    from sklearn.metrics import confusion_matrix, classification_report

    cm = confusion_matrix(all_labels.numpy(), all_predictions.numpy())

    print(f"\n测试结果:")
    print(f"总体准确率: {accuracy:.4f}")
    print(f"\n混淆矩阵:")
    print(f"               预测正常    预测异常")
    print(f"真实正常  |    {cm[0, 0]:>6}    {cm[0, 1]:>6}")
    print(f"真实异常  |    {cm[1, 0]:>6}    {cm[1, 1]:>6}")

    # 计算各类别准确率
    normal_accuracy = cm[0, 0] / cm[0].sum() if cm[0].sum() > 0 else 0
    abnormal_accuracy = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0

    print(f"\n正常样本准确率: {normal_accuracy:.4f}")
    print(f"异常样本准确率: {abnormal_accuracy:.4f}")

    # 打印分类报告
    print(f"\n详细分类报告:")
    print(classification_report(all_labels.numpy(), all_predictions.numpy(),
                                target_names=['正常', '异常'], digits=4))

    # 分析置信度
    normal_indices = all_labels == 0
    abnormal_indices = all_labels == 1

    if normal_indices.any():
        normal_confidences = all_probabilities[normal_indices, 0]
        print(f"\n正常样本置信度:")
        print(f"  平均值: {normal_confidences.mean():.4f}")
        print(f"  标准差: {normal_confidences.std():.4f}")
        print(f"  最小值: {normal_confidences.min():.4f}")
        print(f"  最大值: {normal_confidences.max():.4f}")

    if abnormal_indices.any():
        abnormal_confidences = all_probabilities[abnormal_indices, 1]
        print(f"\n异常样本置信度:")
        print(f"  平均值: {abnormal_confidences.mean():.4f}")
        print(f"  标准差: {abnormal_confidences.std():.4f}")
        print(f"  最小值: {abnormal_confidences.min():.4f}")
        print(f"  最大值: {abnormal_confidences.max():.4f}")

    return {
        'accuracy': accuracy,
        'normal_accuracy': normal_accuracy,
        'abnormal_accuracy': abnormal_accuracy,
        'confusion_matrix': cm,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities
    }


def analyze_decision_boundary(model, test_results):
    """分析门控决策边界"""
    print("\n" + "="*70)
    print("分析门控决策边界")
    print("="*70)

    probabilities = test_results['probabilities']
    labels = test_results['labels']

    # 提取正常和异常样本的预测概率
    normal_probs = probabilities[labels == 0]
    abnormal_probs = probabilities[labels == 1]

    # 分析决策阈值
    thresholds = np.arange(0.1, 1.0, 0.05)

    print(f"\n不同决策阈值下的性能:")
    print(f"{'阈值':<8} {'正常准确率':<12} {'异常准确率':<12} {'总体准确率':<12}")
    print("-" * 50)

    best_threshold = 0.5
    best_overall_accuracy = 0.0

    for threshold in thresholds:
        # 对于正常样本，预测概率 > threshold 被认为是异常
        normal_pred = (normal_probs[:, 0] < threshold).long()
        normal_acc = (normal_pred == 0).float().mean().item()

        # 对于异常样本，预测概率 > threshold 被认为是异常
        abnormal_pred = (abnormal_probs[:, 1] > threshold).long()
        abnormal_acc = abnormal_pred.float().mean().item()

        # 总体准确率
        total_normal = len(normal_probs)
        total_abnormal = len(abnormal_probs)
        overall_accuracy = (normal_acc * total_normal + abnormal_acc * total_abnormal) / (total_normal + total_abnormal)

        print(f"{threshold:<8.2f} {normal_acc:<12.4f} {abnormal_acc:<12.4f} {overall_accuracy:<12.4f}")

        if overall_accuracy > best_overall_accuracy:
            best_overall_accuracy = overall_accuracy
            best_threshold = threshold

    print(f"\n最佳阈值: {best_threshold:.2f}")
    print(f"最佳总体准确率: {best_overall_accuracy:.4f}")

    # 绘制ROC曲线
    from sklearn.metrics import roc_curve, auc

    # 将二分类问题转换为异常检测问题
    y_true = labels.numpy()
    y_scores = probabilities[:, 1].numpy()  # 异常类的概率

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))

    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')

    # 标记最佳阈值
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red',
                s=100, label=f'Optimal Threshold={optimal_threshold:.3f}')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve of Gate Network')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('gate_roc_curve.png', dpi=150, bbox_inches='tight')
    

    print(f"\nROC曲线已保存到 'gate_roc_curve.png'")
    print(f"AUC值: {roc_auc:.4f}")
    print(f"基于Youden指数的最佳阈值: {optimal_threshold:.4f}")

    return {
        'best_threshold': best_threshold,
        'best_overall_accuracy': best_overall_accuracy,
        'roc_auc': roc_auc,
        'optimal_threshold': optimal_threshold
    }


# 主测试函数
def main_test():
    """主测试函数"""

    # 加载模型
    model, checkpoint = load_gate_model()

    if model is None:
        return

    # 测试门控网络性能
    test_results = test_gate_performance(model, test)

    # 分析决策边界
    boundary_analysis = analyze_decision_boundary(model, test_results)


# 运行测试
if __name__ == "__main__":
    main_test()
