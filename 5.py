import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------- 1. 全局配置 --------------------------
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlepad'] = 12


# -------------------------- 2. 工具函数 --------------------------
def create_dir_safe(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"📂 自动创建目录：{dir_path}")
    return dir_path


def check_dependencies():
    required_libs = [
        "pandas", "numpy", "matplotlib", "seaborn",
        "xgboost", "tensorflow", "joblib"
    ]
    missing_libs = []
    for lib in required_libs:
        try:
            __import__(lib)
        except ImportError:
            missing_libs.append(lib)

    if missing_libs:
        print(f"❌ 检测到缺失依赖库：{', '.join(missing_libs)}")
        print(f"👉 请运行以下命令安装：")
        print(f"pip install {' '.join(missing_libs)} -i https://pypi.douban.com/simple/")
        exit(1)
    print("✅ 所有核心依赖库均已安装")


# -------------------------- 3. 数据加载 --------------------------
def load_battery_data(data_path=r"E:\QC\detailed_ev_battery_data.csv"):
    if not os.path.exists(data_path):
        print(f"❌ 数据集文件不存在：{data_path}")
        print("👉 请先运行「详细新能源汽车电池数据集生成器」脚本生成数据")
        return None

    try:
        df = pd.read_csv(data_path, encoding="utf-8-sig")
        print(f"✅ 数据加载成功！数据规模：{df.shape[0]}行 × {df.shape[1]}列")
        print(f"📋 数据列名：{list(df.columns)}")
        return df
    except Exception as e:
        print(f"❌ 数据读取失败：{str(e)}")
        return None


# -------------------------- 4. 数据预处理（修复核心错误） --------------------------
def preprocess_data(df):
    target_col = "SOH实测值(%)"
    drop_cols = [target_col, "日期", "异常情况备注"]
    feature_cols = [col for col in df.columns if col not in drop_cols]

    X = df[feature_cols].copy()
    y = df[target_col].values

    print(f"\n📊 数据拆分完成：")
    print(f" - 特征矩阵(X)：{X.shape}（行：样本数，列：特征数）")
    print(f" - 目标变量(y)：{y.shape}（numpy数组，0-based索引）")

    # 区分类别特征与数值特征
    categorical_cols = [
        "车型", "充电类型", "使用场景", "场景类型",
        "续航变化趋势", "公里数变化趋势"
    ]
    numerical_cols = [col for col in feature_cols if col not in categorical_cols]

    print(f"\n🔍 特征类型区分：")
    print(f" - 类别特征（{len(categorical_cols)}个）：{categorical_cols}")
    print(f" - 数值特征（{len(numerical_cols)}个）：{numerical_cols}")

    # 数值特征标准化
    X_num = X[numerical_cols].values
    num_mean = X_num.mean(axis=0)
    num_std = X_num.std(axis=0)
    num_std[num_std == 0] = 1e-8
    X_num_scaled = (X_num - num_mean) / num_std

    # 类别特征独热编码（修复错误：先保留DataFrame获取列名，再转数组）
    X_cat = X[categorical_cols]
    # 关键修复：先创建DataFrame并保存列名，再转换为numpy数组
    X_cat_encoded_df = pd.get_dummies(X_cat, drop_first=False, dummy_na=False)
    X_cat_encoded = X_cat_encoded_df.values  # 转换为数组
    cat_encoded_cols = list(X_cat_encoded_df.columns)  # 从DataFrame获取列名

    # 合并特征
    X_processed = np.hstack([X_num_scaled, X_cat_encoded])

    # 处理后的特征名称（数值特征 + 编码后的类别特征）
    processed_feature_names = numerical_cols + cat_encoded_cols

    # 划分训练集和测试集
    np.random.seed(42)
    shuffle_indices = np.random.permutation(len(X_processed))
    train_size = int(len(X_processed) * 0.7)
    train_indices = shuffle_indices[:train_size]
    test_indices = shuffle_indices[train_size:]

    X_train = X_processed[train_indices]
    X_test = X_processed[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    # 保存预处理参数
    preprocess_params = {
        "num_mean": num_mean,
        "num_std": num_std,
        "categorical_cols": categorical_cols,
        "numerical_cols": numerical_cols,
        "cat_encoded_cols": cat_encoded_cols  # 使用修复后的列名列表
    }

    print(f"✅ 预处理完成！")
    print(f" - 训练集：{X_train.shape}（样本数 × 处理后特征数）")
    print(f" - 测试集：{X_test.shape}")
    print(f" - 处理后总特征数：{len(processed_feature_names)}")
    return X_train, X_test, y_train, y_test, preprocess_params, processed_feature_names


# -------------------------- 5. 时序数据转换 --------------------------
def create_sequence_data(X_train, X_test, y_train, y_test, sequence_length=6):
    def sliding_window(data_X, data_y, seq_len):
        X_seq = []
        y_seq = []
        for i in range(len(data_X) - seq_len):
            X_seq.append(data_X[i:i + seq_len])
            y_seq.append(data_y[i + seq_len])
        return np.array(X_seq), np.array(y_seq)

    print(f"\n⏳ 生成时序序列数据（窗口长度：{sequence_length}）...")
    X_train_seq, y_train_seq = sliding_window(X_train, y_train, sequence_length)
    X_test_seq, y_test_seq = sliding_window(X_test, y_test, sequence_length)

    print(f"✅ 时序数据生成完成！")
    print(f" - 训练序列：X={X_train_seq.shape} | y={y_train_seq.shape}")
    print(f" - 测试序列：X={X_test_seq.shape} | y={y_test_seq.shape}")
    return X_train_seq, X_test_seq, y_train_seq, y_test_seq


# -------------------------- 6. 模型构建 --------------------------
def build_lstm(input_shape):
    model = Sequential([
        LSTM(units=64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=32),
        Dropout(0.2),
        Dense(units=16, activation="relu"),
        Dense(units=1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def build_gru(input_shape):
    model = Sequential([
        GRU(units=64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(units=32),
        Dropout(0.2),
        Dense(units=16, activation="relu"),
        Dense(units=1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def build_xgboost():
    return xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror"
    )


# -------------------------- 7. 评估指标计算 --------------------------
def calculate_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total) if ss_total != 0 else 0.0
    return round(mse, 4), round(mae, 4), round(r2, 4)


# -------------------------- 8. 模型训练与评估 --------------------------
def train_and_evaluate_models(X_train, X_test, y_train, y_test,
                              X_train_seq, X_test_seq, y_train_seq, y_test_seq):
    models = {}
    results = {}
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    # 训练LSTM模型
    print("\n" + "=" * 50)
    print("📌 1/3 训练LSTM模型")
    print("=" * 50)
    lstm_input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
    lstm_model = build_lstm(lstm_input_shape)

    lstm_history = lstm_model.fit(
        X_train_seq, y_train_seq,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )

    lstm_preds = lstm_model.predict(X_test_seq).flatten()
    lstm_mse, lstm_mae, lstm_r2 = calculate_metrics(y_test_seq, lstm_preds)

    results["LSTM"] = {
        "mse": lstm_mse,
        "mae": lstm_mae,
        "r2": lstm_r2,
        "preds": lstm_preds,
        "actual": y_test_seq,
        "history": lstm_history
    }
    models["LSTM"] = lstm_model
    print(f"✅ LSTM模型训练完成：MSE={lstm_mse} | MAE={lstm_mae} | R2={lstm_r2}")

    # 训练GRU模型
    print("\n" + "=" * 50)
    print("📌 2/3 训练GRU模型")
    print("=" * 50)
    gru_model = build_gru(lstm_input_shape)

    gru_history = gru_model.fit(
        X_train_seq, y_train_seq,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )

    gru_preds = gru_model.predict(X_test_seq).flatten()
    gru_mse, gru_mae, gru_r2 = calculate_metrics(y_test_seq, gru_preds)

    results["GRU"] = {
        "mse": gru_mse,
        "mae": gru_mae,
        "r2": gru_r2,
        "preds": gru_preds,
        "actual": y_test_seq,
        "history": gru_history
    }
    models["GRU"] = gru_model
    print(f"✅ GRU模型训练完成：MSE={gru_mse} | MAE={gru_mae} | R2={gru_r2}")

    # 训练XGBoost模型
    print("\n" + "=" * 50)
    print("📌 3/3 训练XGBoost模型")
    print("=" * 50)
    xgb_model = build_xgboost()

    xgb_model.fit(X_train, y_train)

    xgb_preds = xgb_model.predict(X_test)
    xgb_mse, xgb_mae, xgb_r2 = calculate_metrics(y_test, xgb_preds)

    results["XGBoost"] = {
        "mse": xgb_mse,
        "mae": xgb_mae,
        "r2": xgb_r2,
        "preds": xgb_preds,
        "actual": y_test,
        "model": xgb_model
    }
    models["XGBoost"] = xgb_model
    print(f"✅ XGBoost模型训练完成：MSE={xgb_mse} | MAE={xgb_mae} | R2={xgb_r2}")

    return models, results


# -------------------------- 9. 模型性能对比与可视化 --------------------------
def compare_models(results, save_dir=r"E:\QC\model_visualizations"):
    save_dir = create_dir_safe(save_dir)

    perf_data = []
    for model_name, res in results.items():
        perf_data.append({
            "模型": model_name,
            "MSE（均方误差）": res["mse"],
            "MAE（平均绝对误差）": res["mae"],
            "R2（决定系数）": res["r2"]
        })
    perf_df = pd.DataFrame(perf_data).sort_values("R2（决定系数）", ascending=False)

    print("\n" + "=" * 70)
    print("📊 3种模型性能对比表（按R2降序排列）")
    print("=" * 70)
    print(perf_df.to_string(index=False))
    print("=" * 70)

    # 性能指标对比图
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    sns.barplot(x="模型", y="MSE（均方误差）", data=perf_df, palette="Blues_r")
    plt.title("模型MSE对比（越小越好）", fontsize=12, fontweight="bold")
    plt.xticks(rotation=45)
    for i, v in enumerate(perf_df["MSE（均方误差）"]):
        plt.text(i, v + 0.0002, f"{v}", ha="center", va="bottom")

    plt.subplot(1, 3, 2)
    sns.barplot(x="模型", y="MAE（平均绝对误差）", data=perf_df, palette="Greens_r")
    plt.title("模型MAE对比（越小越好）", fontsize=12, fontweight="bold")
    plt.xticks(rotation=45)
    for i, v in enumerate(perf_df["MAE（平均绝对误差）"]):
        plt.text(i, v + 0.0005, f"{v}", ha="center", va="bottom")

    plt.subplot(1, 3, 3)
    sns.barplot(x="模型", y="R2（决定系数）", data=perf_df, palette="Oranges")
    plt.title("模型R2对比（越接近1越好）", fontsize=12, fontweight="bold")
    plt.xticks(rotation=45)
    plt.ylim(0.95, 1.0)
    for i, v in enumerate(perf_df["R2（决定系数）"]):
        plt.text(i, v + 0.001, f"{v}", ha="center", va="bottom")

    plt.tight_layout()
    perf_plot_path = os.path.join(save_dir, "model_performance_comparison.png")
    plt.savefig(perf_plot_path)
    plt.close()
    print(f"\n✅ 性能对比图保存：{perf_plot_path}")

    # 预测值vs实际值散点图
    plt.figure(figsize=(15, 5))
    for idx, (model_name, res) in enumerate(results.items(), 1):
        plt.subplot(1, 3, idx)

        actual = res["actual"]
        preds = res["preds"]
        sample_size = min(100, len(actual))
        sample_idx = np.random.choice(len(actual), sample_size, replace=False)

        plt.scatter(actual[sample_idx], preds[sample_idx], alpha=0.6, s=30)
        plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], "r--", linewidth=2)
        plt.title(f"{model_name}（R2={res['r2']}）", fontsize=11)
        plt.xlabel("实际SOH值(%)")
        plt.ylabel("预测SOH值(%)")
        plt.grid(alpha=0.3)

    pred_plot_path = os.path.join(save_dir, "prediction_vs_actual.png")
    plt.savefig(pred_plot_path)
    plt.close()
    print(f"✅ 预测对比图保存：{pred_plot_path}")

    # LSTM/GRU训练历史曲线
    for model_name in ["LSTM", "GRU"]:
        if model_name not in results:
            continue
        history = results[model_name]["history"]

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history["loss"], label="训练损失")
        plt.plot(history.history["val_loss"], label="验证损失")
        plt.title(f"{model_name} 损失曲线（MSE）", fontsize=11)
        plt.xlabel("训练轮次")
        plt.ylabel("损失值")
        plt.legend()
        plt.grid(alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(history.history["mae"], label="训练MAE")
        plt.plot(history.history["val_mae"], label="验证MAE")
        plt.title(f"{model_name} MAE曲线", fontsize=11)
        plt.xlabel("训练轮次")
        plt.ylabel("MAE值")
        plt.legend()
        plt.grid(alpha=0.3)

        history_plot_path = os.path.join(save_dir, f"{model_name}_training_history.png")
        plt.savefig(history_plot_path)
        plt.close()
        print(f"✅ {model_name}训练历史图保存：{history_plot_path}")

    best_model_name = perf_df.iloc[0]["模型"]
    best_r2 = perf_df.iloc[0]["R2（决定系数）"]
    print(f"\n🎉 最佳模型评选结果：{best_model_name}（R2={best_r2}）")
    return best_model_name


# -------------------------- 10. 保存最佳模型 --------------------------
def save_best_model(models, best_model_name, preprocess_params, save_dir=r"E:\QC\best_battery_model"):
    save_dir = create_dir_safe(save_dir)
    best_model = models[best_model_name]

    if best_model_name in ["LSTM", "GRU"]:
        model_path = os.path.join(save_dir, f"{best_model_name}_model.h5")
        best_model.save(model_path)
    else:
        model_path = os.path.join(save_dir, f"{best_model_name}_model.pkl")
        joblib.dump(best_model, model_path)

    preprocess_path = os.path.join(save_dir, "preprocess_params.pkl")
    joblib.dump(preprocess_params, preprocess_path)

    usage_info = f"""# 新能源汽车电池健康值预测 - 最佳模型说明
============================
模型名称：{best_model_name}
保存时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
适用场景：电池健康值（SOH）预测
============================

## 文件列表
1. 模型文件：{os.path.basename(model_path)}
2. 预处理参数：{os.path.basename(preprocess_path)}

## 使用步骤
1. 加载预处理参数：
   preprocess_params = joblib.load("preprocess_params.pkl")

2. 加载最佳模型：
   - 若为LSTM/GRU：
     from tensorflow.keras.models import load_model
     model = load_model("{os.path.basename(model_path)}")
   - 若为XGBoost：
     model = joblib.load("{os.path.basename(model_path)}")

3. 新数据预处理：
   a. 提取数值特征并标准化：
      X_num = new_data[preprocess_params['numerical_cols']].values
      X_num_scaled = (X_num - preprocess_params['num_mean']) / preprocess_params['num_std']

   b. 提取类别特征并编码：
      X_cat = new_data[preprocess_params['categorical_cols']]
      X_cat_encoded = pd.get_dummies(X_cat)[preprocess_params['cat_encoded_cols']]
      X_cat_encoded = X_cat_encoded.fillna(0).values

   c. 合并特征：
      X_processed = np.hstack([X_num_scaled, X_cat_encoded])

4. 预测SOH值：
   - 若为LSTM/GRU（需时序数据）：
     soh_pred = model.predict(X_new_seq)
   - 若为XGBoost：
     soh_pred = model.predict(X_processed)
"""
    info_path = os.path.join(save_dir, "model_usage_guide.txt")
    with open(info_path, "w", encoding="utf-8") as f:
        f.write(usage_info)

    print(f"\n📁 最佳模型保存完成！")
    print(f" - 模型文件：{model_path}")
    print(f" - 预处理参数：{preprocess_path}")
    print(f" - 使用说明：{info_path}")


# -------------------------- 11. 主函数 --------------------------
def main():
    print("=" * 70)
    print("🚗 新能源汽车电池健康值(SOH)预测模型训练流程")
    print("=" * 70)

    check_dependencies()
    df = load_battery_data()
    if df is None:
        return

    X_train, X_test, y_train, y_test, preprocess_params, feature_names = preprocess_data(df)
    X_train_seq, X_test_seq, y_train_seq, y_test_seq = create_sequence_data(
        X_train, X_test, y_train, y_test, sequence_length=6
    )
    models, results = train_and_evaluate_models(
        X_train, X_test, y_train, y_test,
        X_train_seq, X_test_seq, y_train_seq, y_test_seq
    )
    best_model_name = compare_models(results)
    save_best_model(models, best_model_name, preprocess_params)

    print("\n" + "=" * 70)
    print("✅ 所有流程执行完成！")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n❌ 流程被用户手动中断")
    except Exception as e:
        print(f"\n❌ 流程执行出错：{str(e)}")
        print("👉 请检查：1. 数据文件是否存在 2. 依赖库是否安装 3. 路径是否有读写权限")
