# Docker Build Performance Guide

## ビルド時間の目安

### キャッシュが効く場合（コード変更のみ）
- **所要時間**: 10-30秒
- **処理内容**: 
  - Pythonコードのコピー
  - ディレクトリ作成
  - 最終レイヤーの作成

### キャッシュが効かない場合

#### requirements.txt 変更時
- **所要時間**: 3-10分
- **処理内容**:
  - 全Pythonパッケージの再インストール
  - PyTorch、OpenCVなど大きなパッケージのダウンロード

#### Dockerfile変更時（システムパッケージ）
- **所要時間**: 5-15分
- **処理内容**:
  - apt-get update
  - システムパッケージのインストール
  - Pythonパッケージのインストール

## キャッシュを最大限活用する方法

### 1. Dockerfileの順序を最適化

```dockerfile
# ❌ 悪い例：コードを先にコピー
COPY . .
RUN pip install -r requirements.txt
# → コード変更のたびにpip installが走る

# ✅ 良い例：requirements.txtを先にコピー
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
# → requirements.txt が変わらない限りpip installはキャッシュ
```

### 2. .dockerignoreを活用

```
# .dockerignore
__pycache__
*.pyc
.git
.pytest_cache
logs/
*.log
docs/
README.md
```

不要なファイルを除外することで：
- ビルドコンテキストが小さくなる
- COPYが高速化
- 不要な変更でキャッシュが無効にならない

### 3. マルチステージビルド（さらなる最適化）

```dockerfile
# ビルドステージ
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# 実行ステージ
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
```

## VPSでのビルド時間短縮テクニック

### 1. ビルドキャッシュの保持
```bash
# 定期的にキャッシュをクリーンしない
# docker system prune -a  # これは避ける
```

### 2. レイヤーキャッシュの共有
```bash
# 前のイメージをベースに使う
docker build --cache-from $DOCKER_IMAGE:latest -t $DOCKER_IMAGE:latest .
```

### 3. requirements.txtの分割
```dockerfile
# 変更頻度の低い依存関係
COPY requirements-base.txt .
RUN pip install -r requirements-base.txt

# 変更頻度の高い依存関係
COPY requirements.txt .
RUN pip install -r requirements.txt
```

## 実測値（2コアVPSの場合）

| シナリオ | キャッシュあり | キャッシュなし |
|---------|--------------|---------------|
| コード変更のみ | 15-30秒 | - |
| 小さなパッケージ追加 | 1-2分 | 5-10分 |
| PyTorch更新 | - | 10-20分 |
| 完全な初回ビルド | - | 15-30分 |

## 推奨事項

1. **開発中**: キャッシュを活用して高速ビルド
2. **本番デプロイ**: 月1回程度は`--no-cache`で完全ビルド（セキュリティアップデート適用）
3. **CI/CD**: ビルドキャッシュを永続化する仕組みを導入