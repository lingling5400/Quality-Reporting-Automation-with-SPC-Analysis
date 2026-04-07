import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


class AutoMLPipeline:
    
    def __init__(self):
        self.df = None
    
    # =====================
    # 1️⃣ 讀資料 + 基本處理
    # =====================
    def read_csv(self, path):
        self.df = pd.read_csv(path)
        
        # rename label
        if "Pass/Fail" in self.df.columns:
            self.df = self.df.rename(columns={"Pass/Fail": "label"})
            print("Before:", self.df['label'].unique())
        
        if "label" in self.df.columns:
            self.df['label'] = self.df['label'].replace(-1, 0)
            print("After:", self.df['label'].unique())
        
        print("原始 shape:", self.df.shape)
        print("缺失值總數:", self.df.isnull().sum().sum())
    
    # =====================
    # 2️⃣ 清理資料
    # =====================
    def drop_high_missing(self, threshold=0.5):
        missing_ratio = self.df.isnull().mean()
        self.df = self.df.loc[:, missing_ratio < threshold]
        
        print("刪除高缺失後 shape:", self.df.shape)
    
    def fill_missing(self):
        self.df = self.df.fillna(self.df.mean(numeric_only=True))
        print("缺失值已填補")
    
    # =====================
    # 3️⃣ 3σ 方法（統計）
    # =====================
    def detect_3sigma(self, sensor_name):
        sensor = self.df[sensor_name]
        
        mean = sensor.mean()
        std = sensor.std()
        ucl = mean + 3 * std
        lcl = mean - 3 * std
        
        pred = ((sensor > ucl) | (sensor < lcl)).astype(int)
        return pred
    
    def evaluate_3sigma(self, sensor_name):
        print(f"\n 3σ 評估 ({sensor_name})")
        
        pred = self.detect_3sigma(sensor_name)
        true = (self.df['label'] == 1).astype(int)
        
        print(classification_report(true, pred))
    
    # =====================
    # 4️⃣ ML 方法
    # =====================
    def train_ml(self):
        print("\n ML 模型 (RandomForest)")
        
        X = self.df.drop(columns=['label', 'Time'], errors='ignore')
        y = self.df['label']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=0.2, random_state=42
        )
        
        model = RandomForestClassifier(class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)
        
        pred = model.predict(X_test)
        
        print(classification_report(y_test, pred))
    
    # =====================
    # 5️⃣ 視覺化（EDA用）
    # =====================
    def plot_sensor(self, sensor_name):
        sensor = self.df[sensor_name]
        
        mean = sensor.mean()
        std = sensor.std()
        ucl = mean + 3 * std
        lcl = mean - 3 * std
        
        plt.figure(figsize=(12,4))
        plt.plot(sensor, label=sensor_name)
        plt.axhline(mean, linestyle='--', label='Mean',color='red')
        plt.axhline(ucl, linestyle='--', label='UCL')
        plt.axhline(lcl, linestyle='--', label='LCL')
        
        plt.legend()
        plt.title(f"{sensor_name} Control Chart")
        plt.show()


# =====================
#  主程式
# =====================
if __name__ == "__main__":
    
    file_path = "C:/Users/user/Downloads/archive/uci-secom.csv"  # 改成自己檔案位置
    
    pipeline = AutoMLPipeline()
    
    # 1. 讀資料
    pipeline.read_csv(file_path)
    
    # 2. 清理
    pipeline.drop_high_missing()
    pipeline.fill_missing()
    
    # 3. 統計方法（挑一個sensor測試）
    example_sensor = pipeline.df.columns[1]  # 隨便抓一個
    pipeline.evaluate_3sigma(example_sensor)
    
    # 4. ML 方法
    pipeline.train_ml()
    
    # 5. 畫圖（選用）
    pipeline.plot_sensor(example_sensor)