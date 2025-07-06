#!/bin/bash

# Script to migrate from master to main branch

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}🔄 Migrating from master to main branch${NC}"
echo "======================================"

# 1. ローカルブランチをmainにリネーム
echo -e "${YELLOW}1. Renaming local branch...${NC}"
git branch -m master main

# 2. 新しいmainブランチをプッシュ
echo -e "${YELLOW}2. Pushing main branch...${NC}"
git push -u origin main

# 3. GitHubのデフォルトブランチをmainに変更する案内
echo ""
echo -e "${YELLOW}3. Change default branch on GitHub:${NC}"
echo "   1. Go to: https://github.com/ktlarc0719/logo-detection-api/settings"
echo "   2. Click on 'Branches' in the left sidebar"
echo "   3. Change default branch from 'master' to 'main'"
echo "   4. Click 'Update'"
echo ""
echo -e "${GREEN}Press Enter after you've changed the default branch on GitHub...${NC}"
read -r

# 4. 古いmasterブランチを削除
echo -e "${YELLOW}4. Delete old master branch? (y/n)${NC}"
read -r delete_master

if [[ $delete_master == "y" || $delete_master == "Y" ]]; then
    echo "Deleting remote master branch..."
    git push origin --delete master
    echo -e "${GREEN}✓ Remote master branch deleted${NC}"
fi

# 5. ローカル設定を更新
echo -e "${YELLOW}5. Updating local configuration...${NC}"
git config init.defaultBranch main

echo ""
echo -e "${GREEN}✅ Migration complete!${NC}"
echo ""
echo "Your default branch is now: main"
echo "Future pushes: git push origin main"
echo ""
echo "To set main as default for all new repositories:"
echo "  git config --global init.defaultBranch main"