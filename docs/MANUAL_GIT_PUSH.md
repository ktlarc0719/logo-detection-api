# Manual Git Push Instructions

ターミナルが落ちる場合は、以下のコマンドを1つずつ手動で実行してください：

## 1. 現在の状態を確認
```bash
cd /root/projects/logo-detection-api
git status
```

## 2. 変更をステージング
```bash
git add .
```

## 3. コミット
```bash
git commit -m "Update project files"
```

## 4. リモートの確認
```bash
git remote -v
```

## 5. プッシュ
```bash
# 既存のブランチの場合
git push origin main

# または新しいブランチの場合
git push -u origin main
```

## トラブルシューティング

### もしgitが初期化されていない場合：
```bash
git init
git add .
git commit -m "Initial commit"
```

### リモートが設定されていない場合：
```bash
git remote add origin https://github.com/YOUR_USERNAME/logo-detection-api.git
```

### 認証エラーの場合：
1. GitHubで Personal Access Token を作成
   - GitHub → Settings → Developer settings → Personal access tokens
   - Generate new token (classic)
   - repo権限にチェック

2. プッシュ時にユーザー名とトークンを使用：
   - Username: YOUR_GITHUB_USERNAME
   - Password: YOUR_PERSONAL_ACCESS_TOKEN

### 大きなファイルがある場合：
```bash
# .gitignoreに追加
echo "*.pt" >> .gitignore
echo "*.onnx" >> .gitignore
echo "models/" >> .gitignore
git rm -r --cached models/
git add .gitignore
git commit -m "Remove large files"
```

## 別の方法：GitHub CLIを使用

1. GitHub CLIをインストール：
```bash
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update
sudo apt install gh
```

2. 認証：
```bash
gh auth login
```

3. リポジトリ作成とプッシュ：
```bash
gh repo create logo-detection-api --private --source=. --push
```