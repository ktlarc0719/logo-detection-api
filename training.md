# labelImgを使用したアノテーション作業とトレーニングガイド

## テストで最低限必要な画像の枚数

### 実験・テスト段階
- **最小限**: 各クラス（ロゴの種類）につき20-30枚でも動作確認可能
- **初期テスト**: 各クラス50-100枚あれば基本的な学習が可能
- **テスト用**: 全体の20%程度を検証用に分ける

### 本番運用向け
- **推奨**: 各クラスにつき200-300枚以上
- **理想的**: 各クラス500枚以上で高精度を期待できる

### 少ない画像で始めるコツ
- データ拡張（augmentation）を積極的に使用
- 事前学習済みモデルを活用（転移学習）
- まず少量で試してから、徐々に画像を追加

### 画像の多様性を確保
- 異なる角度
- 異なる照明条件
- 異なる背景
- 異なるサイズ/解像度

## labelImgの出力と配置

### 出力ファイル
labelImgは以下のファイルを生成します：
- **YOLO形式**: 各画像に対して1つのテキストファイル
- ファイル名: `画像名.txt` (例: `image001.jpg` → `image001.txt`)
- フォーマット: `class_id x_center y_center width height` (すべて正規化された値)

### ディレクトリ構造
```
logo-detection-api/
├── data/
│   ├── images/
│   │   ├── train/         # トレーニング用画像
│   │   └── val/           # 検証用画像
│   ├── labels/
│   │   ├── train/         # トレーニング用ラベル（.txt）
│   │   └── val/           # 検証用ラベル（.txt）
│   └── classes.txt        # クラス名のリスト（1行に1クラス）
```

### labelImgでの作業手順
1. 画像を`data/images/train/`または`data/images/val/`に配置
2. `classes.txt`ファイルを作成し、各行にクラス名を記述
3. labelImgを起動し、**YOLO形式**に切り替える（左メニューから「YOLO」を選択）
4. 画像ディレクトリを開く（Open Dir）
5. Change Save Dir で保存先を対応する`labels/`フォルダに設定
6. 各画像でバウンディングボックスを描画し、ラベルを付ける
7. Ctrl+S（またはCmd+S）で保存

### YOLO形式の例
```
# image001.txt の内容例
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.15 0.2
```
- 各行が1つのオブジェクト
- 値は全て0-1に正規化
- class_id は0から始まるインデックス

## 学習開始時のパラメータ

### 基本的なトレーニングパラメータ

```python
# YOLOv5/YOLOv8の場合
training_params = {
    'epochs': 100,              # エポック数（初期テストは50-100）
    'batch_size': 16,           # バッチサイズ（GPUメモリに応じて調整）
    'img_size': 640,            # 入力画像サイズ
    'lr': 0.01,                 # 学習率
    'momentum': 0.937,          # モメンタム
    'weight_decay': 0.0005,     # 重み減衰
    'conf_thres': 0.25,         # 信頼度の閾値
    'iou_thres': 0.45,          # IoU閾値
}

# データ拡張パラメータ
augmentation_params = {
    'hsv_h': 0.015,            # 色相の変動
    'hsv_s': 0.7,              # 彩度の変動
    'hsv_v': 0.4,              # 明度の変動
    'degrees': 0.0,            # 回転角度
    'translate': 0.1,          # 平行移動
    'scale': 0.5,              # スケール変更
    'shear': 0.0,              # せん断変形
    'perspective': 0.0,        # 透視変換
    'flipud': 0.0,             # 上下反転
    'fliplr': 0.5,             # 左右反転
    'mosaic': 1.0,             # モザイク拡張
    'mixup': 0.0,              # MixUp拡張
}
```

### 小規模データセットでの推奨設定

```python
# データが少ない場合の設定
small_dataset_params = {
    'epochs': 200,              # より多くのエポック
    'batch_size': 8,            # 小さめのバッチサイズ
    'patience': 50,             # Early stoppingの待機エポック数
    'cache': True,              # 画像をメモリにキャッシュ
    'workers': 4,               # データローダーのワーカー数
}
```

### トレーニング開始コマンド例

```bash
# YOLOv5の場合
python train.py \
    --img 640 \
    --batch 16 \
    --epochs 100 \
    --data data/dataset.yaml \
    --weights yolov5s.pt \
    --cache

# カスタムスクリプトの場合
python train_logo_detector.py \
    --train-path data/images/train \
    --val-path data/images/val \
    --annotations data/annotations \
    --output models/logo_detector \
    --epochs 100
```

## データセット設定ファイル

YOLOトレーニング用の`data/dataset.yaml`を作成：

```yaml
# dataset.yaml
path: ../data  # データセットのルートパス
train: images/train  # トレーニング画像のパス
val: images/val      # 検証画像のパス

# クラス
nc: 2  # クラス数
names: ['logo1', 'logo2']  # クラス名のリスト
```

## 注意事項

1. **クラスの一貫性**: すべてのアノテーションで同じクラス名を使用
2. **品質チェック**: アノテーション後、.txtファイルを確認（値が0-1の範囲内か）
3. **バランス**: 各クラスの画像数をできるだけ均等に
4. **検証セット**: 必ず独立した検証データを用意
5. **ファイル対応**: 各画像ファイルに対応する同名の.txtファイルが必要