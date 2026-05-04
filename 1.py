import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# -------------------------- 新增：解决中文显示问题 --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 适配Windows中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# --------------------------------------------------------------------------

# 1. 加载数据集（确保路径与你的本地路径一致）
file_path = "C:/Users/Lenovo/PycharmProjects/pythonProject13/毕业设计/tesla_10-20wkm_diverse_data.csv"
df = pd.read_csv(file_path)

# 2. 特征与目标变量选择
features = [
    "累计行驶里程(km)", "当日行驶里程(km)", "初始电池容量(kWh)",
    "环境温度(℃)", "使用场景", "振动频率(Hz)", "振动时长占比(%)",
    "充电类型", "充电倍率(C)", "充电电压(V)", "充电时间(h)",
    "单体电压最大差异(V)", "充放电循环次数", "三因素耦合衰减系数"
]
target = "SOH实测值(%)"
X = df[features]
y = df[target]

# 3. 数据划分（7:3）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 4. 特征预处理管道（类别特征独热编码）
numeric_features = [col for col in features if X[col].dtype in [np.int64, np.float64]]
categorical_features = [col for col in features if X[col].dtype == object]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 5. 模型训练（随机森林 + XGBoost）
# 5.1 随机森林
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=100, max_depth=10, min_samples_split=10, random_state=42, n_jobs=-1
    ))
])
rf_pipeline.fit(X_train, y_train)

# 5.2 XGBoost
xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
    ))
])
xgb_pipeline.fit(X_train, y_train)

# 6. 模型评估函数
def evaluate_model(model, X, y, model_name):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f"--- {model_name} 评估结果 ---")
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"均方根误差 (RMSE): {rmse:.4f}")
    print(f"平均绝对误差 (MAE): {mae:.4f}")
    print(f"决定系数 (R²): {r2:.4f}\n")
    return y_pred

# 输出评估结果
rf_y_pred_test = evaluate_model(rf_pipeline, X_test, y_test, "随机森林（测试集）")
xgb_y_pred_test = evaluate_model(xgb_pipeline, X_test, y_test, "XGBoost（测试集）")

# 7. 可视化（中文正常显示）
plt.figure(figsize=(14, 6))

# 7.1 预测值vs真实值（XGBoost，最优模型）
plt.subplot(1, 2, 1)
plt.scatter(y_test, xgb_y_pred_test, alpha=0.5, color='#FF6B6B', label='预测值')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', linewidth=2, label='理想拟合线')
plt.xlabel('真实SOH值(%)', fontsize=11)
plt.ylabel('预测SOH值(%)', fontsize=11)
plt.title('XGBoost：预测值 vs 真实值', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

# 7.2 残差分布（误差分析）
plt.subplot(1, 2, 2)
residuals = xgb_y_pred_test - y_test
sns.histplot(residuals, kde=True, color='#4ECDC4', bins=50)
plt.xlabel('残差（预测值-真实值）', fontsize=11)
plt.ylabel('频率', fontsize=11)
plt.title('XGBoost残差分布（均值≈0）', fontsize=12, fontweight='bold')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("C:/Users/Lenovo/PycharmProjects/pythonProject13/毕业设计/SOH预测结果图.png", dpi=300, bbox_inches='tight')
plt.show()

# 8. 保存最优模型（XGBoost），方便后续部署
import joblib
joblib.dump(xgb_pipeline, "C:/Users/Lenovo/PycharmProjects/pythonProject13/毕业设计/SOH预测_XGBoost模型.pkl")
print("✅ 最优模型已保存为：SOH预测_XGBoost模型.pkl")