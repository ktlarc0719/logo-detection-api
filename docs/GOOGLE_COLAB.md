# Google Colab でのロゴ検出モデル開発ガイド

## 1. 環境セットアップ

### 1.1 必要なライブラリのインストール

!pip install ultralytics
!pip install roboflow
!pip install tensorboard

### 1.2 Google Driveのマウント

from google.colab import drive
drive.mount('/content/drive')

### 1.3 作業ディレクトリの作成と移動

import os

# 作業ディレクトリの作成
work_dir = '/content/drive/MyDrive/logo_detection'
os.makedirs(work_dir, exist_ok=True)
os.chdir(work_dir)

print(f"Current working directory: {os.getcwd()}")

### 1.4 GPU環境の確認

import torch
from ultralytics import YOLO

### 1.5 GPU情報の表示
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # GPU使用率の確認
    print(f"\nCurrent GPU memory usage:")
    print(f"Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")


## 2. データセットの準備

### 2.1 Roboflowからデータセットをダウンロード

from roboflow import Roboflow

# Roboflow APIキーでログイン
rf = Roboflow(api_key="YOUR_API_KEY")

# プロジェクトとデータセットを取得
project = rf.workspace("YOUR_WORKSPACE").project("logo-detection")
version = project.version(1)

# YOLOv8形式でダウンロード
dataset = version.download("yolov8", location="./datasets/logo_dataset")

print(f"Dataset downloaded to: {dataset.location}")
print(f"Number of images: {len(dataset.train)} train, {len(dataset.valid)} val, {len(dataset.test)} test")


### 2.2 手動でデータセットを準備する場合

import shutil

# データセット構造の作成
dataset_dir = './datasets/custom_dataset'
for split in ['train', 'val', 'test']:
    os.makedirs(f'{dataset_dir}/images/{split}', exist_ok=True)
    os.makedirs(f'{dataset_dir}/labels/{split}', exist_ok=True)

# dataset.yamlファイルの作成
dataset_yaml = """
# YOLOv8 Dataset Configuration
path: ../datasets/custom_dataset  # dataset root dir
train: images/train  # train images
val: images/val      # val images
test: images/test    # test images (optional)

# Classes
names:
  0: logo  # ロゴクラス
  # 複数のブランドを区別する場合
  # 0: nike
  # 1: adidas
  # 2: apple
  # 3: google

# Number of classes
nc: 1
"""

with open(f'{dataset_dir}/dataset.yaml', 'w', encoding='utf-8') as f:
    f.write(dataset_yaml)

print(f"Dataset structure created at: {dataset_dir}")


### 2.3 データセットの検証

import cv2
import matplotlib.pyplot as plt
import numpy as np

def visualize_dataset(dataset_path, num_samples=5):
    """データセットのサンプルを可視化"""
    
    # dataset.yamlの読み込み
    import yaml
    with open(f'{dataset_path}/dataset.yaml', 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    # 画像とラベルのパスを取得
    images_dir = f"{dataset_path}/{dataset_config['train']}"
    labels_dir = images_dir.replace('images', 'labels')
    
    # 画像ファイルのリスト
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
    
    # サンプル表示
    fig, axes = plt.subplots(1, min(num_samples, len(image_files)), figsize=(20, 4))
    if num_samples == 1:
        axes = [axes]
    
    for idx, img_file in enumerate(image_files[:num_samples]):
        # 画像の読み込み
        img_path = os.path.join(images_dir, img_file)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # ラベルの読み込み
        label_path = os.path.join(labels_dir, img_file.replace('.jpg', '.txt').replace('.png', '.txt'))
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, x_center, y_center, width, height = map(float, parts)
                        
                        # YOLO形式から画素座標に変換
                        x1 = int((x_center - width/2) * w)
                        y1 = int((y_center - height/2) * h)
                        x2 = int((x_center + width/2) * w)
                        y2 = int((y_center + height/2) * h)
                        
                        # バウンディングボックスの描画
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img, f'Class {int(class_id)}', (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        axes[idx].imshow(img)
        axes[idx].set_title(f'{img_file}')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()

# データセットの可視化
visualize_dataset('./dataset', num_samples=5)


## 3. モデルのトレーニング

### 3.1 トレーニングパラメータの設定とモデルの訓練

from ultralytics import YOLO

# モデルの選択
# yolov8n.pt: Nano (最速・最軽量)
# yolov8s.pt: Small (高速・軽量)
# yolov8m.pt: Medium (バランス型)
# yolov8l.pt: Large (高精度)
# yolov8x.pt: Extra Large (最高精度)

model = YOLO('yolov8s.pt')  # 事前学習済みモデルをロード

# トレーニングの実行
results = model.train(
    data='datasets/custom_dataset/dataset.yaml',  # データセット設定ファイル
    epochs=100,                    # エポック数
    imgsz=640,                     # 入力画像サイズ
    batch=16,                      # バッチサイズ（GPUメモリに応じて調整）
    name='logo_detector_v1',       # 実験名
    patience=20,                   # Early Stoppingの忍耐度
    save=True,                     # チェックポイントの保存
    device=0,                      # GPU使用（0: 最初のGPU）
    verbose=True,                  # 詳細ログ出力
    
    # データ拡張設定
    hsv_h=0.015,                   # 色相の変動（0-1）
    hsv_s=0.7,                     # 彩度の変動（0-1）
    hsv_v=0.4,                     # 明度の変動（0-1）
    degrees=0.0,                   # 回転角度（-180 to +180）
    translate=0.1,                 # 平行移動（0-1）
    scale=0.5,                     # スケール変動（0-1）
    shear=0.0,                     # せん断変形（-180 to +180）
    perspective=0.0,               # 透視変換（0-0.001）
    flipud=0.0,                    # 上下反転確率（0-1）
    fliplr=0.5,                    # 左右反転確率（0-1）
    mosaic=1.0,                    # モザイク拡張確率（0-1）
    mixup=0.0,                     # MixUp拡張確率（0-1）
    copy_paste=0.0,                # Copy-Paste拡張確率（0-1）
    
    # 最適化設定
    optimizer='SGD',               # オプティマイザー（'SGD', 'Adam', 'AdamW'）
    lr0=0.01,                      # 初期学習率
    lrf=0.01,                      # 最終学習率（lr0 * lrf）
    momentum=0.937,                # SGDモメンタム/Adamベータ1
    weight_decay=0.0005,           # 重み減衰
    warmup_epochs=3.0,             # ウォームアップエポック数
    warmup_momentum=0.8,           # ウォームアップ初期モメンタム
    warmup_bias_lr=0.1,            # ウォームアップ初期バイアス学習率
    
    # その他の高度な設定
    close_mosaic=10,               # モザイクデータ拡張を無効にする最終エポック
    amp=True,                      # 自動混合精度トレーニング
    fraction=1.0,                  # トレーニングデータの使用割合
    profile=False,                 # プロファイリングの有効化
    overlap_mask=True,             # トレーニング中のマスクオーバーラップ（セグメント）
    mask_ratio=4,                  # マスクダウンサンプリング比率（セグメント）
    
    # 検証設定
    val=True,                      # トレーニング中の検証
    plots=True,                    # トレーニングプロットの保存
)

print("Training completed!")
print(f"Best model saved at: {results.save_dir}/weights/best.pt")


### 3.2 トレーニング進捗のリアルタイム監視

# TensorBoardでの監視
%load_ext tensorboard
%tensorboard --logdir runs/detect

# トレーニング中の損失値を確認
import pandas as pd

# CSVファイルから結果を読み込み
results_csv = pd.read_csv('runs/detect/logo_detector_v1/results.csv')

# 最新の指標を表示
print("\nLatest Training Metrics:")
print(f"Epoch: {results_csv['epoch'].iloc[-1]}")
print(f"Train Box Loss: {results_csv['train/box_loss'].iloc[-1]:.4f}")
print(f"Train Class Loss: {results_csv['train/cls_loss'].iloc[-1]:.4f}")
print(f"Val Box Loss: {results_csv['val/box_loss'].iloc[-1]:.4f}")
print(f"Val Class Loss: {results_csv['val/cls_loss'].iloc[-1]:.4f}")
print(f"mAP50: {results_csv['metrics/mAP50(B)'].iloc[-1]:.4f}")
print(f"mAP50-95: {results_csv['metrics/mAP50-95(B)'].iloc[-1]:.4f}")


### 3.3 トレーニング結果の可視化

from IPython.display import Image, display
import matplotlib.pyplot as plt

# トレーニング結果の表示
print("Training Results:")
display(Image('runs/detect/logo_detector_v1/results.png'))

# 混同行列の表示
print("\nConfusion Matrix:")
display(Image('runs/detect/logo_detector_v1/confusion_matrix.png'))

# PR曲線の表示
print("\nPR Curve:")
display(Image('runs/detect/logo_detector_v1/PR_curve.png'))

# F1曲線の表示
print("\nF1 Curve:")
display(Image('runs/detect/logo_detector_v1/F1_curve.png'))

# 検証バッチの例
print("\nValidation Batch Examples:")
display(Image('runs/detect/logo_detector_v1/val_batch0_pred.jpg'))


## 4. モデルの評価

### 4.1 検証データでの詳細評価

# 最良モデルのロード
model = YOLO('runs/detect/logo_detector_v1/weights/best.pt')

# 検証データでの評価
metrics = model.val(
    data='datasets/custom_dataset/dataset.yaml',
    batch=16,
    imgsz=640,
    conf=0.001,  # 信頼度閾値
    iou=0.6,     # NMS IoU閾値
    device=0,    # GPU使用
)

# 評価結果の表示
print("Validation Results:")
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")

# クラス別の結果
for i, class_name in enumerate(metrics.names.values()):
    print(f"\nClass '{class_name}':")
    print(f"  AP50: {metrics.box.ap50[i]:.4f}")
    print(f"  AP: {metrics.box.ap[i]:.4f}")


### 4.2 推論速度のベンチマーク

import time

# ベンチマーク設定
num_runs = 100
img_size = 640

# ダミー画像で推論速度を測定
dummy_img = torch.randn(1, 3, img_size, img_size).cuda()

# ウォームアップ
for _ in range(10):
    _ = model(dummy_img, verbose=False)

# 速度測定
torch.cuda.synchronize()
start_time = time.time()

for _ in range(num_runs):
    _ = model(dummy_img, verbose=False)

torch.cuda.synchronize()
total_time = time.time() - start_time

# 結果表示
avg_time = total_time / num_runs
fps = 1 / avg_time

print(f"Average inference time: {avg_time*1000:.2f} ms")
print(f"FPS: {fps:.2f}")
print(f"Total time for {num_runs} runs: {total_time:.2f} seconds")


## 5. モデルの推論とテスト

### 5.1 単一画像での推論

# テスト画像での推論
test_image_path = 'path/to/test/image.jpg'

# 推論の実行
results = model(
    test_image_path,
    conf=0.25,      # 信頼度閾値
    iou=0.45,       # NMS IoU閾値
    imgsz=640,      # 推論画像サイズ
    device=0,       # GPU使用
)

# 結果の表示
for r in results:
    # 検出結果の表示
    print(f"Number of detections: {len(r.boxes)}")
    
    # 各検出の詳細
    for i, box in enumerate(r.boxes):
        cls = int(box.cls)
        conf = float(box.conf)
        xyxy = box.xyxy[0].tolist()
        
        print(f"\nDetection {i+1}:")
        print(f"  Class: {model.names[cls]}")
        print(f"  Confidence: {conf:.4f}")
        print(f"  Bounding box: {xyxy}")
    
    # 画像の保存と表示
    annotated_img = r.plot()
    cv2.imwrite('detection_result.jpg', annotated_img)
    
    # Colabで表示
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Detection Results')
    plt.show()


### 5.2 バッチ推論

# 複数画像での推論
test_images_dir = 'path/to/test/images'
image_files = [f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.png'))]

# バッチ推論の実行
results = model(
    [os.path.join(test_images_dir, f) for f in image_files],
    conf=0.25,
    iou=0.45,
    imgsz=640,
    device=0,
    stream=True,  # ストリーミングモード
)

# 結果の処理
detection_summary = []
for idx, r in enumerate(results):
    detections = []
    for box in r.boxes:
        detections.append({
            'class': model.names[int(box.cls)],
            'confidence': float(box.conf),
            'bbox': box.xyxy[0].tolist()
        })
    
    detection_summary.append({
        'image': image_files[idx],
        'detections': detections,
        'num_detections': len(detections)
    })

# サマリーの表示
print(f"Total images processed: {len(detection_summary)}")
print(f"Images with detections: {sum(1 for d in detection_summary if d['num_detections'] > 0)}")
print(f"Total detections: {sum(d['num_detections'] for d in detection_summary)}")


### 5.3 複数クラスのテストデータ評価

# 複数クラスのテストデータを評価する場合
import os
from collections import defaultdict

# モデルのロード
model = YOLO('runs/detect/logo_detector_v1/weights/best.pt')

# クラス名の定義（dataset.yamlと同じ順序で）
class_names = ['nike', 'adidas', 'apple', 'google']  # 例

# 各クラスのテスト画像を収集
test_results = defaultdict(lambda: {'total': 0, 'detected': 0, 'correct': 0})

# 各クラスのテストフォルダから画像を処理
for class_idx, class_name in enumerate(class_names):
    test_dir = f'dataset/test/{class_name}/images'
    
    if not os.path.exists(test_dir):
        print(f"Warning: {test_dir} not found")
        continue
    
    # このクラスの画像を取得
    image_files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png'))]
    test_results[class_name]['total'] = len(image_files)
    
    print(f"\nProcessing {class_name} class ({len(image_files)} images)...")
    
    # バッチ推論
    image_paths = [os.path.join(test_dir, f) for f in image_files]
    results = model(image_paths, conf=0.25, iou=0.45, device=0)
    
    # 結果の集計
    for r in results:
        if len(r.boxes) > 0:
            test_results[class_name]['detected'] += 1
            
            # 検出されたクラスが正しいかチェック
            detected_classes = [int(box.cls) for box in r.boxes]
            if class_idx in detected_classes:
                test_results[class_name]['correct'] += 1

# 結果の表示
print("\n=== Test Results by Class ===")
print(f"{'Class':<15} {'Total':<10} {'Detected':<10} {'Correct':<10} {'Precision':<10}")
print("-" * 55)

overall_total = 0
overall_detected = 0
overall_correct = 0

for class_name, stats in test_results.items():
    total = stats['total']
    detected = stats['detected']
    correct = stats['correct']
    precision = correct / detected if detected > 0 else 0
    
    print(f"{class_name:<15} {total:<10} {detected:<10} {correct:<10} {precision:<10.2%}")
    
    overall_total += total
    overall_detected += detected
    overall_correct += correct

# 全体の結果
print("-" * 55)
overall_precision = overall_correct / overall_detected if overall_detected > 0 else 0
print(f"{'Overall':<15} {overall_total:<10} {overall_detected:<10} {overall_correct:<10} {overall_precision:<10.2%}")


### 5.4 混同行列を使った詳細評価

# より詳細な評価（混同行列）
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 予測結果と正解ラベルを収集
y_true = []
y_pred = []

for class_idx, class_name in enumerate(class_names):
    test_dir = f'dataset/test/{class_name}/images'
    
    if not os.path.exists(test_dir):
        continue
    
    image_files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png'))]
    image_paths = [os.path.join(test_dir, f) for f in image_files]
    
    # 推論
    results = model(image_paths, conf=0.25, iou=0.45, device=0)
    
    for r in results:
        # 正解ラベル
        y_true.append(class_idx)
        
        # 予測ラベル（最も信頼度の高いもの）
        if len(r.boxes) > 0:
            confidences = [box.conf.item() for box in r.boxes]
            classes = [int(box.cls) for box in r.boxes]
            best_idx = np.argmax(confidences)
            y_pred.append(classes[best_idx])
        else:
            y_pred.append(-1)  # 未検出

# 混同行列の作成
cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

# 可視化
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, 
            yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# クラスごとの精度計算
for i, class_name in enumerate(class_names):
    tp = cm[i, i]
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n{class_name}:")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1-Score: {f1:.3f}")


### 5.5 動画での推論

# 動画ファイルでの推論
video_path = 'path/to/test/video.mp4'
output_path = 'output_video.mp4'

# 動画の読み込み
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 出力動画の設定
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# フレームごとの処理
frame_count = 0
detection_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 推論
    results = model(frame, conf=0.25, iou=0.45)
    
    # 結果の描画
    annotated_frame = results[0].plot()
    out.write(annotated_frame)
    
    # 統計情報の更新
    frame_count += 1
    detection_count += len(results[0].boxes)
    
    # 進捗表示（10フレームごと）
    if frame_count % 10 == 0:
        print(f"Processed {frame_count} frames, Total detections: {detection_count}")

# リソースの解放
cap.release()
out.release()

print(f"\nVideo processing completed!")
print(f"Total frames: {frame_count}")
print(f"Total detections: {detection_count}")
print(f"Average detections per frame: {detection_count/frame_count:.2f}")
print(f"Output saved to: {output_path}")


## 6. モデルのエクスポートとデプロイ

### 6.1 様々な形式へのエクスポート

# ONNXフォーマットへのエクスポート
model.export(
    format='onnx',
    imgsz=640,
    simplify=True,
    opset=12,
    dynamic=False,
    batch=1,
)

# TensorRTへのエクスポート（NVIDIA GPU環境のみ）
model.export(
    format='engine',
    imgsz=640,
    device=0,
    half=True,  # FP16精度
    workspace=4,  # GiB
)

# CoreML（iOS向け）へのエクスポート
model.export(
    format='coreml',
    imgsz=640,
    nms=True,
)

# TFLite（モバイル向け）へのエクスポート
model.export(
    format='tflite',
    imgsz=640,
    int8=False,  # INT8量子化
)

print("Model exported successfully!")


### 6.2 エクスポートしたモデルのテスト

# ONNXモデルのテスト
import onnxruntime as ort
import numpy as np

# ONNXセッションの作成
onnx_model_path = 'runs/detect/logo_detector_v1/weights/best.onnx'
ort_session = ort.InferenceSession(onnx_model_path)

# 入力の準備
test_img = cv2.imread('path/to/test/image.jpg')
test_img = cv2.resize(test_img, (640, 640))
test_img = test_img.transpose(2, 0, 1)  # HWC to CHW
test_img = test_img.astype(np.float32) / 255.0
test_img = np.expand_dims(test_img, axis=0)

# 推論
outputs = ort_session.run(None, {'images': test_img})
print(f"ONNX output shape: {outputs[0].shape}")


### 6.3 モデルの保存と共有

import zipfile
import datetime

# モデルと設定をパッケージング
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
package_name = f"logo_detector_{timestamp}.zip"

with zipfile.ZipFile(package_name, 'w') as zipf:
    # モデルファイル
    zipf.write('runs/detect/logo_detector_v1/weights/best.pt', 'model/best.pt')
    zipf.write('runs/detect/logo_detector_v1/weights/best.onnx', 'model/best.onnx')
    
    # 設定ファイル
    zipf.write('datasets/custom_dataset/dataset.yaml', 'config/dataset.yaml')
    zipf.write('runs/detect/logo_detector_v1/args.yaml', 'config/training_args.yaml')
    
    # 結果
    zipf.write('runs/detect/logo_detector_v1/results.png', 'results/training_curves.png')
    zipf.write('runs/detect/logo_detector_v1/confusion_matrix.png', 'results/confusion_matrix.png')

print(f"Model package created: {package_name}")

# Google Driveに保存
shutil.copy(package_name, f"/content/drive/MyDrive/logo_detection/models/{package_name}")
print(f"Model saved to Google Drive")


## 7. 継続的な改善

### 7.1 追加トレーニング（Fine-tuning）

# 既存モデルの追加トレーニング
model = YOLO('runs/detect/logo_detector_v1/weights/best.pt')

# 新しいデータで追加トレーニング
results = model.train(
    data='datasets/new_dataset/dataset.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='logo_detector_v2_finetune',
    resume=False,  # False: 新しいトレーニング、True: 中断からの再開
    freeze=10,     # 最初の10層を凍結
)


### 7.2 ハイパーパラメータチューニング

# グリッドサーチの例
hyperparams = {
    'lr0': [0.001, 0.01, 0.1],
    'batch': [8, 16, 32],
    'epochs': [50, 100],
    'imgsz': [640, 768]
}

results_summary = []

for lr in hyperparams['lr0']:
    for batch in hyperparams['batch']:
        model = YOLO('yolov8s.pt')
        results = model.train(
            data='datasets/custom_dataset/dataset.yaml',
            epochs=50,  # 短縮版
            imgsz=640,
            batch=batch,
            lr0=lr,
            name=f'tune_lr{lr}_batch{batch}',
            patience=10
        )
        
        # 結果の記録
        metrics = model.val()
        results_summary.append({
            'lr0': lr,
            'batch': batch,
            'mAP50': metrics.box.map50,
            'mAP50-95': metrics.box.map
        })

# 結果の表示
import pandas as pd
df_results = pd.DataFrame(results_summary)
print(df_results.sort_values('mAP50-95', ascending=False))


### 7.3 モデルの監視とメンテナンス

# 推論ログの記録
import json
from datetime import datetime

def log_inference(image_path, predictions, model_version):
    """推論結果をログに記録"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'model_version': model_version,
        'image': image_path,
        'predictions': predictions,
        'num_detections': len(predictions)
    }
    
    # ログファイルに追記
    with open('inference_log.jsonl', 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

# 性能モニタリング
def monitor_performance(log_file='inference_log.jsonl'):
    """推論ログから性能指標を計算"""
    logs = []
    with open(log_file, 'r') as f:
        for line in f:
            logs.append(json.loads(line))
    
    df = pd.DataFrame(logs)
    
    # 日別の統計
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    daily_stats = df.groupby('date').agg({
        'num_detections': ['mean', 'std', 'count']
    })
    
    print("Daily Detection Statistics:")
    print(daily_stats)
    
    return daily_stats


## まとめ

このガイドでは、Google Colabを使用してYOLOv8ベースのロゴ検出モデルを開発する完全なワークフローを説明しました。

主なステップ：
1. 環境のセットアップとGPU確認
2. データセットの準備と検証
3. モデルのトレーニングと監視
4. 評価とベンチマーク
5. 推論とテスト
6. エクスポートとデプロイ
7. 継続的な改善

次のステップ：
- より多くのロゴクラスの追加
- データ拡張戦略の改善
- モデルアンサンブルの実装
- リアルタイム推論の最適化
- エッジデバイスへのデプロイ