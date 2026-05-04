import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# -------------------------- 1. 配置与工具函数 --------------------------
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300


def create_dir_safe(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"📂 创建目录：{dir_path}")
    return dir_path


def check_dependencies():
    required_libs = ["pandas", "numpy", "matplotlib", "seaborn", "xgboost", "tensorflow", "sklearn"]
    missing_libs = []
    for lib in required_libs:
        try:
            __import__(lib)
        except ImportError:
            missing_libs.append(lib)
    if missing_libs:
        print(f"❌ 缺失库：{', '.join(missing_libs)}")
        print(f"👉 安装：pip install {' '.join(missing_libs)} -i https://pypi.douban.com/simple/")
        exit(1)
    print("✅ 依赖检查通过")


# -------------------------- 2. 数据加载与基础预处理 --------------------------
def load_data(data_path=r"E:\QC\detailed_ev_battery_data.csv"):
    if not os.path.exists(data_path):
        print(f"❌ 数据不存在：{data_path}")
        return None
    try:
        df = pd.read_csv(data_path, encoding="utf-8-sig")
        print(f"✅ 数据加载成功：{df.shape}")
        # 确保日期列是字符串类型（避免解析错误）
        if "日期" in df.columns:
            df["日期"] = df["日期"].astype(str)
        return df.sort_values(by=["车型", "日期"]).reset_index(drop=True)
    except Exception as e:
        print(f"❌ 加载失败：{e}")
        return None


# -------------------------- 3. 增强版特征工程（修复数据类型） --------------------------
def feature_engineering(df):
    df_new = df.copy()
    numerical_cols = [
        "初始电池容量(kWh)", "当前电池容量(kWh)", "实时百公里能耗(kWh/100km)",
        "当前续航里程(km)", "环境温度(℃)", "振动频率(Hz)", "累计行驶里程(km)",
        "充放电循环次数", "充电时间(h)"
    ]
    categorical_cols = ["车型", "充电类型", "使用场景"]

    # 1. 确保数值列是float类型（核心修复1）
    for col in numerical_cols:
        df_new[col] = pd.to_numeric(df_new[col], errors="coerce").fillna(0)  # 强制转换为数值，缺失值填0

    # 2. 按车型分组计算时序特征
    for col in numerical_cols:
        # 滑动窗口统计
        df_new[f"{col}_3次均值"] = df_new.groupby("车型")[col].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        ).fillna(0)  # 填充可能的NaN
        df_new[f"{col}_5次均值"] = df_new.groupby("车型")[col].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        ).fillna(0)
        df_new[f"{col}_3次标准差"] = df_new.groupby("车型")[col].transform(
            lambda x: x.rolling(window=3, min_periods=1).std().fillna(0)
        )

        # 趋势特征
        df_new[f"{col}_变化量"] = df_new.groupby("车型")[col].diff().fillna(0)

    # 3. 交互特征
    df_new["车型_能耗交互"] = df_new["车型"].astype("category").cat.codes * df_new["实时百公里能耗(kWh/100km)"]
    df_new["充电类型_循环次数交互"] = df_new["充电类型"].astype("category").cat.codes * df_new["充放电循环次数"]

    # 4. 强制所有新特征为float类型（核心修复2）
    new_features = [col for col in df_new.columns if col not in df.columns]
    for col in new_features:
        df_new[col] = df_new[col].astype(float)

    print(f"✅ 特征工程完成：新增{len(new_features)}个特征，总特征数{df_new.shape[1]}")
    return df_new, numerical_cols, categorical_cols


# -------------------------- 4. 数据预处理（强化类型转换） --------------------------
def preprocess_data(df, numerical_cols, categorical_cols):
    target_col = "SOH实测值(%)"
    drop_cols = [target_col, "日期", "异常情况备注", "场景类型", "续航变化趋势", "公里数变化趋势"]
    feature_cols = [col for col in df.columns if col not in drop_cols]

    X = df[feature_cols].copy()
    y = df[target_col].values.astype(float)  # 确保目标值是float

    # 类别特征编码
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    encoded_cat_cols = [col for col in X.columns if any(cat in col for cat in categorical_cols)]

    # 数值特征标准化
    num_cols = [col for col in X.columns if col not in encoded_cat_cols]
    X_num = X[num_cols].values.astype(float)  # 强制转换为float
    num_mean, num_std = X_num.mean(axis=0), X_num.std(axis=0)
    num_std[num_std == 0] = 1e-8
    X[num_cols] = (X_num - num_mean) / num_std

    # 强制所有列都是float32（TensorFlow最兼容的类型，核心修复3）
    X = X.astype(np.float32)

    # 划分训练集和测试集
    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    preprocess_params = {
        "num_mean": num_mean, "num_std": num_std,
        "num_cols": num_cols, "encoded_cat_cols": encoded_cat_cols,
        "feature_cols": feature_cols
    }
    print(f"✅ 预处理完成：训练集{X_train.shape}，测试集{X_test.shape}")
    print(f"   数据类型检查：X_train.dtypes={X_train.dtypes.unique()}")  # 验证是否全为float
    return X_train, X_test, y_train, y_test, preprocess_params


# -------------------------- 5. 时序数据转换（确保张量兼容性） --------------------------
def create_sequences(X, y, seq_len=6):
    X_seq, y_seq, indices = [], [], []
    for i in range(seq_len, len(X)):
        # 提取序列并转换为float32（核心修复4）
        seq = X.iloc[i - seq_len:i].values.astype(np.float32)
        X_seq.append(seq)
        y_seq.append(y[i].astype(np.float32))
        indices.append(X.index[i])
    return np.array(X_seq), np.array(y_seq), np.array(indices)


# -------------------------- 6. 模型定义（保持不变） --------------------------
def build_optimized_lstm(input_shape):
    model = Sequential([
        LSTM(32, input_shape=input_shape, return_sequences=False),
        BatchNormalization(),
        Dropout(0.1),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )
    return model


def build_optimized_gru(input_shape):
    model = Sequential([
        GRU(32, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.1),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )
    return model


def build_tuned_xgboost():
    return xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        objective="reg:squarederror"
    )


# -------------------------- 7. 交叉验证训练（保持不变） --------------------------
def cross_validate_models(X_train, y_train, preprocess_params, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=False)
    seq_len = 6
    cv_results = {
        "XGBoost": {"mse": [], "mae": [], "r2": []},
        "LSTM": {"mse": [], "mae": [], "r2": []},
        "GRU": {"mse": [], "mae": [], "r2": []}
    }

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"\n📌 交叉验证 Fold {fold + 1}/{n_splits}")
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        # XGBoost
        xgb_model = build_tuned_xgboost()
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=0)
        xgb_preds = xgb_model.predict(X_val)
        xgb_mse = mean_squared_error(y_val, xgb_preds)
        xgb_mae = mean_absolute_error(y_val, xgb_preds)
        xgb_r2 = r2_score(y_val, xgb_preds)
        cv_results["XGBoost"]["mse"].append(xgb_mse)
        cv_results["XGBoost"]["mae"].append(xgb_mae)
        cv_results["XGBoost"]["r2"].append(xgb_r2)
        print(f"XGBoost：MSE={xgb_mse:.4f} | MAE={xgb_mae:.4f} | R2={xgb_r2:.4f}")

        # 时序模型
        X_tr_seq, y_tr_seq, _ = create_sequences(X_tr, y_tr, seq_len)
        X_val_seq, y_val_seq, _ = create_sequences(X_val, y_val, seq_len)
        input_shape = (X_tr_seq.shape[1], X_tr_seq.shape[2])

        # LSTM
        lstm_model = build_optimized_lstm(input_shape)
        lstm_callback = [
            EarlyStopping(patience=3, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=2)
        ]
        lstm_model.fit(
            X_tr_seq, y_tr_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=30, batch_size=16,
            callbacks=lstm_callback, verbose=0
        )
        lstm_preds = lstm_model.predict(X_val_seq).flatten()
        lstm_mse = mean_squared_error(y_val_seq, lstm_preds)
        lstm_mae = mean_absolute_error(y_val_seq, lstm_preds)
        lstm_r2 = r2_score(y_val_seq, lstm_preds)
        cv_results["LSTM"]["mse"].append(lstm_mse)
        cv_results["LSTM"]["mae"].append(lstm_mae)
        cv_results["LSTM"]["r2"].append(lstm_r2)
        print(f"LSTM：MSE={lstm_mse:.4f} | MAE={lstm_mae:.4f} | R2={lstm_r2:.4f}")

        # GRU
        gru_model = build_optimized_gru(input_shape)
        gru_model.fit(
            X_tr_seq, y_tr_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=30, batch_size=16,
            callbacks=lstm_callback, verbose=0
        )
        gru_preds = gru_model.predict(X_val_seq).flatten()
        gru_mse = mean_squared_error(y_val_seq, gru_preds)
        gru_mae = mean_absolute_error(y_val_seq, gru_preds)
        gru_r2 = r2_score(y_val_seq, gru_preds)
        cv_results["GRU"]["mse"].append(gru_mse)
        cv_results["GRU"]["mae"].append(gru_mae)
        cv_results["GRU"]["r2"].append(gru_r2)
        print(f"GRU：MSE={gru_mse:.4f} | MAE={gru_mae:.4f} | R2={gru_r2:.4f}")

    # 计算均值
    for model in cv_results:
        cv_results[model]["mse_mean"] = np.mean(cv_results[model]["mse"])
        cv_results[model]["mae_mean"] = np.mean(cv_results[model]["mae"])
        cv_results[model]["r2_mean"] = np.mean(cv_results[model]["r2"])

    print("\n" + "=" * 50)
    print("📊 交叉验证平均结果")
    print("=" * 50)
    for model in cv_results:
        print(
            f"{model}：平均MSE={cv_results[model]['mse_mean']:.4f} | 平均MAE={cv_results[model]['mae_mean']:.4f} | 平均R2={cv_results[model]['r2_mean']:.4f}")
    return cv_results


# -------------------------- 8. 最终模型训练与融合（保持不变） --------------------------
def train_final_models(X_train, X_test, y_train, y_test, preprocess_params):
    seq_len = 6
    X_train_seq, y_train_seq, _ = create_sequences(X_train, y_train, seq_len)
    X_test_seq, y_test_seq, test_indices = create_sequences(X_test, y_test, seq_len)
    input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])

    # XGBoost
    xgb_model = build_tuned_xgboost()
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=0)
    xgb_test_preds = xgb_model.predict(X_test.loc[test_indices])

    # LSTM
    lstm_model = build_optimized_lstm(input_shape)
    lstm_callback = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=2)
    ]
    lstm_model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_test_seq, y_test_seq),
        epochs=30, batch_size=16,
        callbacks=lstm_callback, verbose=1
    )
    lstm_test_preds = lstm_model.predict(X_test_seq).flatten()

    # GRU
    gru_model = build_optimized_gru(input_shape)
    gru_model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_test_seq, y_test_seq),
        epochs=30, batch_size=16,
        callbacks=lstm_callback, verbose=1
    )
    gru_test_preds = gru_model.predict(X_test_seq).flatten()

    # 融合
    blend_preds = 0.7 * xgb_test_preds + 0.15 * lstm_test_preds + 0.15 * gru_test_preds
    blend_mse = mean_squared_error(y_test_seq, blend_preds)
    blend_mae = mean_absolute_error(y_test_seq, blend_preds)
    blend_r2 = r2_score(y_test_seq, blend_preds)

    final_results = {
        "XGBoost": {"preds": xgb_test_preds, "mse": mean_squared_error(y_test_seq, xgb_test_preds),
                    "mae": mean_absolute_error(y_test_seq, xgb_test_preds), "r2": r2_score(y_test_seq, xgb_test_preds)},
        "LSTM": {"preds": lstm_test_preds, "mse": mean_squared_error(y_test_seq, lstm_test_preds),
                 "mae": mean_absolute_error(y_test_seq, lstm_test_preds), "r2": r2_score(y_test_seq, lstm_test_preds)},
        "GRU": {"preds": gru_test_preds, "mse": mean_squared_error(y_test_seq, gru_test_preds),
                "mae": mean_absolute_error(y_test_seq, gru_test_preds), "r2": r2_score(y_test_seq, gru_test_preds)},
        "融合模型": {"preds": blend_preds, "mse": blend_mse, "mae": blend_mae, "r2": blend_r2}
    }

    print("\n" + "=" * 50)
    print("📊 测试集最终结果")
    print("=" * 50)
    for model in final_results:
        print(
            f"{model}：MSE={final_results[model]['mse']:.4f} | MAE={final_results[model]['mae']:.4f} | R2={final_results[model]['r2']:.4f}")

    return {
        "models": {"XGBoost": xgb_model, "LSTM": lstm_model, "GRU": gru_model},
        "results": final_results,
        "y_test": y_test_seq,
        "test_indices": test_indices
    }


# -------------------------- 9. 结果可视化 --------------------------
def visualize_results(final_output, save_dir=r"E:\QC\optimized_visuals"):
    save_dir = create_dir_safe(save_dir)
    results = final_output["results"]
    y_test = final_output["y_test"]

    # 性能对比图
    perf_data = []
    for model in results:
        perf_data.append({
            "模型": model,
            "MSE": results[model]["mse"],
            "MAE": results[model]["mae"],
            "R2": results[model]["r2"]
        })
    perf_df = pd.DataFrame(perf_data).sort_values("R2", ascending=False)

    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    sns.barplot(x="模型", y="MSE", data=perf_df, palette="Blues_r")
    plt.title("MSE对比（越小越好）")
    plt.xticks(rotation=45)
    plt.subplot(1, 3, 2)
    sns.barplot(x="模型", y="MAE", data=perf_df, palette="Greens_r")
    plt.title("MAE对比（越小越好）")
    plt.xticks(rotation=45)
    plt.subplot(1, 3, 3)
    sns.barplot(x="模型", y="R2", data=perf_df, palette="Oranges")
    plt.title("R2对比（越近1越好）")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "final_performance.png"))
    plt.close()

    # 预测值对比
    plt.figure(figsize=(12, 8))
    sample_idx = np.random.choice(len(y_test), min(100, len(y_test)), replace=False)
    for i, model in enumerate(results):
        plt.subplot(2, 2, i + 1)
        plt.scatter(y_test[sample_idx], results[model]["preds"][sample_idx], alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
        plt.title(f"{model}（R2={results[model]['r2']:.4f}）")
        plt.xlabel("实际SOH")
        plt.ylabel("预测SOH")
        plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "final_pred_vs_actual.png"))
    plt.close()
    print(f"\n✅ 可视化结果保存至：{save_dir}")


# -------------------------- 10. 保存最佳模型 --------------------------
def save_best_model(final_output, preprocess_params, save_dir=r"E:\QC\best_optimized_model"):
    save_dir = create_dir_safe(save_dir)
    results = final_output["results"]
    models = final_output["models"]

    best_model_name = max(results, key=lambda x: results[x]["r2"])
    best_model = models[best_model_name] if best_model_name != "融合模型" else None

    print(f"\n🎉 最佳模型：{best_model_name}（R2={results[best_model_name]['r2']:.4f}）")

    if best_model_name in ["LSTM", "GRU"]:
        model_path = os.path.join(save_dir, f"{best_model_name}_model.h5")
        best_model.save(model_path)
    elif best_model_name == "XGBoost":
        model_path = os.path.join(save_dir, "XGBoost_model.pkl")
        joblib.dump(best_model, model_path)

    joblib.dump(preprocess_params, os.path.join(save_dir, "preprocess_params.pkl"))
    joblib.dump({"blend_weights": [0.7, 0.15, 0.15]}, os.path.join(save_dir, "blend_weights.pkl"))

    print(f"📁 模型保存至：{save_dir}")


# -------------------------- 11. 主函数 --------------------------
def main():
    print("=" * 70)
    print("🚗 电池SOH预测模型优化训练流程")
    print("=" * 70)

    check_dependencies()
    df = load_data()
    if df is None:
        return

    df_fe, numerical_cols, categorical_cols = feature_engineering(df)
    X_train, X_test, y_train, y_test, preprocess_params = preprocess_data(
        df_fe, numerical_cols, categorical_cols
    )

    cv_results = cross_validate_models(X_train, y_train, preprocess_params)
    final_output = train_final_models(X_train, X_test, y_train, y_test, preprocess_params)
    visualize_results(final_output)
    save_best_model(final_output, preprocess_params)

    print("\n" + "=" * 70)
    print("✅ 优化训练流程完成！")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 错误：{str(e)}")
        print("👉 请检查数据格式或依赖版本")
