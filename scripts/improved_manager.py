"""
Improved manager.py with smart rebuild detection and graceful deployment
"""

@app.route("/git/pull", methods=["POST"])
def git_pull():
    """Pull latest code and intelligently decide whether to rebuild/restart"""
    results = {}
    
    # Pull latest changes
    if not os.path.exists(GIT_REPO_DIR):
        results["clone"] = run_command(f"git clone {GIT_REPO_URL} {GIT_REPO_DIR}")
    else:
        # 現在のコミットを記録
        old_commit = run_command(f"cd {GIT_REPO_DIR} && git rev-parse HEAD")["stdout"].strip()
        
        # Pull実行
        results["pull"] = run_command(f"cd {GIT_REPO_DIR} && git pull origin main")
        
        # 新しいコミット
        new_commit = run_command(f"cd {GIT_REPO_DIR} && git rev-parse HEAD")["stdout"].strip()
        
        if old_commit == new_commit:
            return jsonify({
                "message": "Already up to date",
                "commit": new_commit
            }), 200
        
        # 変更されたファイルをチェック
        diff_result = run_command(
            f"cd {GIT_REPO_DIR} && git diff --name-only {old_commit} {new_commit}"
        )
        changed_files = diff_result["stdout"].strip().split("\n") if diff_result["stdout"].strip() else []
        
        # 再ビルドが必要か判定
        rebuild_needed = any(
            any(pattern in file for pattern in ["src/", "requirements.txt", "Dockerfile"])
            for file in changed_files
        )
        
        # 設定変更のみか判定
        config_only = all(
            file.endswith(('.md', '.txt', '.yml', '.yaml', '.json')) 
            or file.startswith('docs/')
            for file in changed_files
        )
        
        results["analysis"] = {
            "changed_files": changed_files,
            "rebuild_needed": rebuild_needed,
            "config_only": config_only
        }
        
        # ユーザーの指定（デフォルト: True）
        rebuild_flag = request.json.get("rebuild", True) if request.json else True
        
        if force_rebuild or rebuild_needed:
            # ビルド実行
            print("Rebuilding Docker image...")
            results["build"] = run_command(f"cd {GIT_REPO_DIR} && docker build -t {DOCKER_IMAGE} .")
            
            if results["build"]["success"]:
                # グレースフルな再起動
                results["deploy"] = graceful_restart()
        
        elif not config_only:
            # 設定ファイル以外の変更があるが再ビルドは不要な場合
            # （この状況は通常発生しないはず）
            results["warning"] = "Changes detected but rebuild not triggered"
        
        # 変更履歴を保存
        with open("/opt/logo-detection/.last_pull", "w") as f:
            f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "old_commit": old_commit,
                "new_commit": new_commit,
                "changed_files": changed_files,
                "action_taken": "rebuild" if force_rebuild or rebuild_needed else "none"
            }))
    
    return jsonify({
        "success": True,
        "results": results
    }), 200

def graceful_restart():
    """コンテナをグレースフルに再起動"""
    results = {}
    
    # 1. 新しいコンテナを別ポートで起動
    temp_port = 8002
    docker_cmd = [
        "docker", "run", "-d",
        "--name", f"{CONTAINER_NAME}-new",
        "-p", f"{temp_port}:8000",
        "-v", "/opt/logo-detection/logs:/app/logs",
        "-v", "/opt/logo-detection/data:/app/data"
    ]
    
    env_vars = load_env()
    for key, value in env_vars.items():
        docker_cmd.extend(["-e", f"{key}={value}"])
    docker_cmd.append(DOCKER_IMAGE)
    
    results["start_new"] = run_command(docker_cmd)
    
    # 2. 新しいコンテナのヘルスチェック
    time.sleep(5)
    for i in range(30):
        health = run_command(f"curl -s http://localhost:{temp_port}/api/v1/health")
        if health["success"]:
            break
        time.sleep(1)
    
    # 3. 古いコンテナにグレースフルシャットダウン信号を送信
    results["graceful_stop"] = run_command(["docker", "kill", "--signal=SIGTERM", CONTAINER_NAME])
    
    # 4. 30秒待つ（処理中のリクエスト完了待ち）
    time.sleep(30)
    
    # 5. 古いコンテナを削除
    results["remove_old"] = run_command(["docker", "rm", "-f", CONTAINER_NAME])
    
    # 6. 新しいコンテナをリネーム
    results["rename"] = run_command(["docker", "rename", f"{CONTAINER_NAME}-new", CONTAINER_NAME])
    
    # 7. ポートを正しいものに変更（再起動）
    results["final_restart"] = run_command(["docker", "restart", CONTAINER_NAME])
    
    return results