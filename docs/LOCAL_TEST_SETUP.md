# Google Colabで学習したモデルのローカルテスト設定

## 必要なファイル

Google Colabから以下のファイルをダウンロードしてください：

### 1. モデルファイル（必須）
```
runs/detect/logo_detector_v1/weights/best.pt
```

### 2. データセット設定ファイル（推奨）
```
datasets/dataset.yaml
```

## ローカルでのフォルダ構造

```
/root/projects/logo-detection-api/
├── models/
│   └── colab_trained/
│       └── best.pt           # Colabからダウンロードしたモデル
├── test_images/              # テスト用画像を配置
│   ├── nike/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   ├── adidas/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── ...
└── test_colab_model.py      # テストスクリプト
```

## セットアップ手順

### 1. モデル用ディレクトリを作成
```bash
mkdir -p models/colab_trained
mkdir -p test_images
```

### 2. Google Colabからファイルをダウンロード

Google Colabで以下を実行：
```python
# モデルファイルをダウンロード
from google.colab import files
files.download('runs/detect/logo_detector_v1/weights/best.pt')

# または、Google Driveに保存してからダウンロード
import shutil
shutil.copy('runs/detect/logo_detector_v1/weights/best.pt', '/content/drive/MyDrive/best.pt')
```

### 3. ダウンロードしたファイルを配置
```bash
# モデルファイルを配置
mv ~/Downloads/best.pt /root/projects/logo-detection-api/models/colab_trained/

# テスト画像を配置（例）
cp -r ~/your_test_images/* /root/projects/logo-detection-api/test_images/
```

## テストスクリプトの作成

`test_colab_model.py`を作成：

```python
from ultralytics import YOLO
import os
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

# モデルのロード
model_path = 'models/colab_trained/best.pt'
model = YOLO(model_path)

# モデル情報の確認
print(f"Model loaded from: {model_path}")
print(f"Classes: {model.names}")
print(f"Number of classes: {len(model.names)}")

# 1. 単一画像のテスト
def test_single_image(image_path):
    results = model(image_path, conf=0.25, iou=0.45)
    
    for r in results:
        print(f"\nImage: {image_path}")
        print(f"Detections: {len(r.boxes)}")
        
        for box in r.boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            print(f"  - Class: {model.names[cls]}, Confidence: {conf:.3f}")
        
        # 結果を表示
        annotated = r.plot()
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(os.path.basename(image_path))
        plt.show()

# 2. フォルダ内の全画像をテスト
def test_folder(folder_path):
    image_files = list(Path(folder_path).glob('**/*.jpg')) + \
                  list(Path(folder_path).glob('**/*.png'))
    
    results_summary = {}
    
    for img_path in image_files:
        results = model(str(img_path), conf=0.25, iou=0.45)
        
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    'class': model.names[int(box.cls)],
                    'confidence': float(box.conf)
                })
        
        results_summary[str(img_path)] = detections
    
    return results_summary

# 3. クラスごとの評価
def evaluate_by_class(test_dir='test_images'):
    from collections import defaultdict
    
    class_results = defaultdict(lambda: {'total': 0, 'detected': 0, 'correct': 0})
    
    # 各クラスのフォルダを処理
    for class_name in os.listdir(test_dir):
        class_path = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        
        images = list(Path(class_path).glob('*.jpg')) + \
                 list(Path(class_path).glob('*.png'))
        
        class_results[class_name]['total'] = len(images)
        
        for img_path in images:
            results = model(str(img_path), conf=0.25, iou=0.45)
            
            for r in results:
                if len(r.boxes) > 0:
                    class_results[class_name]['detected'] += 1
                    
                    # 検出されたクラスをチェック
                    detected_classes = [model.names[int(box.cls)] for box in r.boxes]
                    if class_name in detected_classes:
                        class_results[class_name]['correct'] += 1
                    break
    
    # 結果を表示
    print("\n=== Evaluation Results ===")
    print(f"{'Class':<15} {'Total':<10} {'Detected':<10} {'Correct':<10} {'Accuracy':<10}")
    print("-" * 60)
    
    for class_name, stats in class_results.items():
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"{class_name:<15} {stats['total']:<10} {stats['detected']:<10} "
              f"{stats['correct']:<10} {accuracy:<10.2%}")

# 実行例
if __name__ == "__main__":
    # 単一画像のテスト
    # test_single_image('test_images/nike/test1.jpg')
    
    # フォルダ全体のテスト
    # results = test_folder('test_images')
    # print(f"Total images processed: {len(results)}")
    
    # クラスごとの評価
    evaluate_by_class('test_images')
```

## 実行方法

```bash
# 仮想環境をアクティベート
source venv/bin/activate

# 必要なライブラリがインストールされていることを確認
pip install ultralytics

# テストを実行
python test_colab_model.py
```

## トラブルシューティング

### 1. クラス名が合わない場合
モデルのクラス名を確認：
```python
model = YOLO('models/colab_trained/best.pt')
print(model.names)  # {0: 'logo', 1: 'nike', ...} のように表示される
```

### 2. 画像サイズの問題
トレーニング時と同じ画像サイズを使用：
```python
results = model(image_path, imgsz=640)  # トレーニング時のサイズに合わせる
```

### 3. 信頼度閾値の調整
検出が少ない/多い場合は閾値を調整：
```python
results = model(image_path, conf=0.1)  # より低い閾値で検出を増やす
```

## APIでの使用

モデルをAPIで使用する場合：

1. モデルファイルを`models/`ディレクトリに配置
2. `.env`ファイルに追加するか、ファイル名でアクセス
3. `/api/v1/models/switch`エンドポイントでモデルを切り替え

```bash
# APIでモデルを切り替え
curl -X POST "http://localhost:8000/api/v1/models/switch?model_name=colab_trained"
```