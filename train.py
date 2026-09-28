import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

def create_training_data(n_samples=6000):
    np.random.seed(42)
    duration = np.random.exponential(scale=1.5, size=n_samples)
    src_bytes = np.random.exponential(scale=1500, size=n_samples)
    dst_bytes = np.random.exponential(scale=3000, size=n_samples)
    conn_count = np.random.poisson(lam=18, size=n_samples)
    srv_count = np.random.poisson(lam=14, size=n_samples)
    same_srv_rate = np.random.uniform(0.1, 1.0, size=n_samples)
    diff_srv_rate = np.random.uniform(0.0, 0.6, size=n_samples)

    # Risk anomaly score
    risk = (conn_count * 0.25) + (src_bytes > 2200) * 2.2 + (diff_srv_rate > 0.35) * 3.5
    label = (risk > np.percentile(risk, 70)).astype(int)

    return pd.DataFrame({
        "Duration": duration,
        "Src_Bytes": src_bytes,
        "Dst_Bytes": dst_bytes,
        "Connection_Count": conn_count,
        "Srv_Count": srv_count,
        "Same_Srv_Rate": same_srv_rate,
        "Diff_Srv_Rate": diff_srv_rate,
        "Label": label
    })

def main():
    print("[1/3] Generating benchmark dataset...")
    df = create_training_data()
    X = df.drop(columns=["Label"])
    y = df["Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("[2/3] Normalizing and training Random Forest Classifier...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_train_scaled, y_train)

    acc = accuracy_score(y_test, clf.predict(X_test_scaled))
    print(f"[+] Model Training Complete! Benchmark Accuracy: {acc * 100:.2f}%")

    print("[3/3] Exporting pipeline artifacts...")
    joblib.dump(clf, "claver_ids_model.pkl")
    joblib.dump(scaler, "claver_ids_scaler.pkl")
    df.to_csv("claver_ids_logs.csv", index=False)
    print("[+] Successfully generated: claver_ids_model.pkl, claver_ids_scaler.pkl, claver_ids_logs.csv")

if __name__ == "__main__":
    main()