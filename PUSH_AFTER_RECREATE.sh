#!/bin/bash
# GitHubリポジトリ再作成後に実行するスクリプト

echo "リモートを再設定しています..."
git remote remove origin
git remote add origin https://github.com/YuZhao20/DEA_model.git

echo "プッシュしています..."
git push -u origin main

echo "完了しました！"

