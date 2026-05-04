import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 准备特征和目标变量
# 选择可能影响SOH值、百公里能耗的特征
features = ['初始电池容量(kWh)', '当前电池容量(kWh)', '环境温度(℃)', '振动频率(Hz)', '振动时长占比(%)',
            '充电倍率(C)', '充电电压(V)', '充电时间(h)', '累计行驶里程(km)', '当日行驶里程(km)',
            '单体电压最大差异(V)', '充放电循环次数', '三因素耦合衰减系数']

# 预测SOH值
X_soh = df[features]
y_soh = df['SOH实测值(%)']

# 划分训练集和测试集
X_soh_train, X_soh_test, y_soh_train, y_soh_test = train_test_split(X_soh, y_soh, test_size=0.2, random_state=42)

# 创建随机森林回归模型
rf_soh = RandomForestRegressor(n_estimators=100, random_state=42)

# 训练模型
rf_soh.fit(X_soh_train, y_soh_train)

# 预测SOH值
y_soh_pred = rf_soh.predict(X_soh_test)

# 评估模型
mse_soh = mean_squared_error(y_soh_test, y_soh_pred)
r2_soh = r2_score(y_soh_test, y_soh_pred)

# 预测百公里能耗
X_consumption = df[features]
y_consumption = df['实时百公里能耗(kWh/100km)']

# 划分训练集和测试集
X_consumption_train, X_consumption_test, y_consumption_train, y_consumption_test = train_test_split(X_consumption, y_consumption, test_size=0.2, random_state=42)

# 创建随机森林回归模型
rf_consumption = RandomForestRegressor(n_estimators=100, random_state=42)

# 训练模型
rf_consumption.fit(X_consumption_train, y_consumption_train)

# 预测百公里能耗
y_consumption_pred = rf_consumption.predict(X_consumption_test)

# 评估模型
mse_consumption = mean_squared_error(y_consumption_test, y_consumption_pred)
r2_consumption = r2_score(y_consumption_test, y_consumption_pred)

# 预测电池全生命周期累计行驶公里数
# 筛选出续航里程小于300公里的数据
end_of_life_data = df[df['当前续航里程(km)'] < 300]

if not end_of_life_data.empty:
    # 取平均累计行驶里程作为估计值
    avg_end_of_life_km = end_of_life_data['累计行驶里程(km)'].mean()
else:
    # 如果没有续航里程小于300公里的数据，假设根据充放电循环次数估算
    # 假设每次循环平均行驶里程，这里简单取数据集中当日行驶里程的平均值
    avg_daily_km = df['当日行驶里程(km)'].mean()
    avg_end_of_life_km = 2000 * avg_daily_km

print(f'SOH值预测的均方误差: {mse_soh:.2f}')
print(f'SOH值预测的R2分数: {r2_soh:.2f}')
print(f'百公里能耗预测的均方误差: {mse_consumption:.2f}')
print(f'百公里能耗预测的R2分数: {r2_consumption:.2f}')
print(f'预计电池报废时累计行驶公里数: {avg_end_of_life_km:.2f}')