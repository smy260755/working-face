import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# 读取 Excel 文件
data = pd.read_excel('../data/604_06.xlsx')

# 获取 x0 和 y0 数据列
time = data['mdt'].values
x0 = data['x0'].values
y0 = data['y0'].values
print(len(x0))

# 进行中值滤波
Out1 = signal.medfilt(x0, 7)  # 滤波计算
Out1 = np.round(Out1, 2)  # 保留两位小数
Out2 = signal.medfilt(y0, 7)
Out2 = np.round(Out2, 2)  # 保留两位小数

print(len(Out1))

# 创建一个新的 DataFrame，包含滤波后的数据
new_data = pd.DataFrame({'mdt': time, 'Out1': Out1, 'Out2': Out2})

# 将新的 DataFrame 保存到新的 xlsx 文件中
new_data.to_excel('new.xlsx', index=False)

# 计算评价指标
def calculate_metrics(original, filtered):
    mse = np.mean((original - filtered) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(original - filtered))
    mape = np.mean(np.abs((original - filtered) / original)) * 100
    return mse, rmse, mae, mape

# 计算 x0 和 y0 的评价指标
mse_x0, rmse_x0, mae_x0, mape_x0 = calculate_metrics(x0, Out1)
mse_y0, rmse_y0, mae_y0, mape_y0 = calculate_metrics(y0, Out2)

print(f"x0 的评价指标：MSE={mse_x0:.2f}, RMSE={rmse_x0:.2f}, MAE={mae_x0:.2f}, MAPE={mape_x0:.2f}%")
print(f"y0 的评价指标：MSE={mse_y0:.2f}, RMSE={rmse_y0:.2f}, MAE={mae_y0:.2f}, MAPE={mape_y0:.2f}%")

# 可视化对比原始数据与滤波后的数据
plt.figure(figsize=(14, 6))

# 绘制原始数据
plt.subplot(1, 2, 1)
plt.plot(time, x0, label='Original x0')
plt.plot(time, y0, label='Original y0')
plt.title('Original data')
plt.xlabel('mdt')
plt.ylabel('Values')
plt.legend()

# 绘制滤波后的数据
plt.subplot(1, 2, 2)
plt.plot(time, Out1, label='Filtered x0', color='orange')
plt.plot(time, Out2, label='Filtered y0', color='green')
plt.title('Filtered data')
plt.xlabel('mdt')
plt.ylabel('Values')
plt.legend()

plt.tight_layout()
plt.show()