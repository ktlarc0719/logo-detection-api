# Cleanup Summary

## 実施日: 2025-07-12

### 削除したファイル

#### テストスクリプト (28個)
- `analyze_and_improve_workflow.py`
- `analyze_training_failure.py`
- `check_confusion_matrix.py`
- `check_dataset_split.py`
- `check_dataset_split_fixed.py`
- `check_model_classes.py`
- `check_training_csv.py`
- `check_training_images.py`
- `check_yolo_training_bug.py`
- `colab_validation_fixed.py`
- `corrected_training_v30.py`
- `debug_training_logs.py`
- `deep_label_investigation.py`
- `diagnose_training_issue.py`
- `example_usage.py`
- `fix_and_retrain.py`
- `fixed_training_script.py`
- `fixed_validation_script.py`
- `handle_imbalanced_multiclass.py`
- `improve_nintendo_performance.py`
- `retrain_with_both_classes.py`
- `training_config_analysis.py`
- `streamlit_app.py`
- その他のテストスクリプト

#### 不要なディレクトリ
- `ui/` - 古いUI実装
- `web_ui/` - 古いWeb UI
- `api/` - 古いAPIディレクトリ（中身は移動済み）
- `venv/` - 仮想環境
- `__pycache__/` - Pythonキャッシュ（全て）

#### その他のファイル
- `simple_performance_test_*.json` - テスト結果
- `build.log` - ビルドログ
- `app.py` - 古いエントリーポイント

### 移動したファイル

#### ドキュメント → `docs/`
- `DEPLOYMENT.md`
- `DOCKER_HUB_DEPLOYMENT.md`
- `LOCAL_TEST_SETUP.md`
- `MANUAL_GIT_PUSH.md`
- `URL_BATCH_API.md`
- `README_INSPECTION.md` → `INSPECTION_FEATURE.md`
- `README_ML_SYSTEM.md` → `ML_SYSTEM_GUIDE.md`
- `README_UI_NAVIGATION.md` → `UI_NAVIGATION_GUIDE.md`
- `training.md`

#### コードファイル
- `api/dataset_splitter.py` → `src/api/endpoints/dataset_splitter.py`

#### モデルファイル
- `yolov8n.pt` → `models/yolov8n.pt`

### 追加したファイル
- `PROJECT_STRUCTURE.md` - プロジェクト構造の説明
- `docs/index.md` - ドキュメントインデックス
- `CLEANUP_SUMMARY.md` - このファイル
- `run_server.sh` - サーバー起動スクリプト

### コード修正
- `src/api/main.py` - importパスの修正
- `src/core/ml_model_visualizer.py` - seaborn依存の削除
- `src/core/ml_model_validator.py` - seaborn依存の削除
- `requirements.txt` - seabornの削除

### 現在のディレクトリ構造
```
logo-detection-api/
├── src/          # ソースコード
├── templates/    # HTMLテンプレート
├── static/       # 静的ファイル
├── models/       # 学習済みモデル
├── datasets/     # データセット
├── docs/         # ドキュメント
├── scripts/      # スクリプト
├── tests/        # テストコード
├── docker/       # Docker関連
├── nginx/        # Nginx設定
├── logs/         # ログ（.gitignore）
├── runs/         # YOLO実行結果（.gitignore）
└── data/         # データ（.gitignore）
```

### 成果
- ルートディレクトリのPythonファイル: 28個 → 0個
- ルートディレクトリのMDファイル: 10個 → 3個（README.md, PROJECT_STRUCTURE.md, CLEANUP_SUMMARY.md）
- 全体的にクリーンで整理された構造
- 依存関係の最適化（不要なseabornを削除）