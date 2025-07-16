# サーバー起動ガイド

## 前提条件
仮想環境（venv）がアクティベートされていることを確認してください。

```bash
# 仮想環境をアクティベート
source venv/bin/activate
```

## 起動方法

### 方法1: uvicornコマンドを直接使用
```bash
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 方法2: start_server.pyを使用
```bash
python start_server.py
```

### 方法3: run_server.shスクリプトを使用（自動的に仮想環境を作成・アクティベート）
```bash
./run_server.sh
```

## トラブルシューティング

### 1. ModuleNotFoundError
必要なパッケージがインストールされていない場合：
```bash
pip install -r requirements.txt
```

### 2. PyTorchエラー
PyTorchのCPU版を再インストール：
```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3. 最小限の依存関係でテスト
基本的な依存関係のみをインストール：
```bash
pip install fastapi uvicorn[standard] python-multipart aiofiles httpx psutil pyyaml pandas matplotlib python-dotenv aiohttp tabulate colorlog
```

### 4. ポートが使用中
別のポートを使用：
```bash
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8001
```

## アクセス方法
サーバーが起動したら、以下のURLでアクセス可能：
- メインダッシュボード: http://localhost:8000/
- APIドキュメント: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 推奨事項
1. 初回起動時は必要最小限のパッケージから始める
2. PyTorchはCPU版を使用（開発環境では十分）
3. GPU版が必要な場合は別途インストール