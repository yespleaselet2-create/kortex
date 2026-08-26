import os
import time
import json
import subprocess
import requests


def push_to_github(repo_name, token, commit_msg="Update checkpoint"):
    try:
        result = subprocess.run(
            ["git", "add", "-A"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            print(f"  git add failed: {result.stderr}")
            return False

        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=30
        )
        if not result.stdout.strip():
            print("  No changes to commit.")
            return False

        result = subprocess.run(
            ["git", "commit", "-m", commit_msg],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            print(f"  git commit failed: {result.stderr}")
            return False

        result = subprocess.run(
            ["git", "push"],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            print(f"  git push failed: {result.stderr}")
            return False

        print("  Pushed to GitHub successfully.")
        return True
    except Exception as e:
        print(f"  GitHub push error: {e}")
        return False


def create_github_repo(repo_name, token, description="Kortex - Code Language Model", private=False):
    api_url = "https://api.github.com/user/repos"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    data = {
        "name": repo_name,
        "description": description,
        "private": private,
        "auto_init": False,
    }

    try:
        resp = requests.post(api_url, headers=headers, json=data, timeout=30)
        if resp.status_code == 201:
            print(f"  Created repo: {resp.json()['html_url']}")
            return resp.json()["clone_url"]
        elif resp.status_code == 422:
            print(f"  Repo already exists, getting URL...")
            resp2 = requests.get(f"https://api.github.com/repos/{token.split(':')[0] if ':' in token else 'user'}/{repo_name}", headers=headers, timeout=30)
            return resp2.json().get("clone_url")
        else:
            print(f"  Failed to create repo: {resp.status_code} {resp.text[:200]}")
            return None
    except Exception as e:
        print(f"  Error creating repo: {e}")
        return None


def push_to_huggingface(local_path, repo_name, token, model_name="kortex"):
    try:
        from huggingface_hub import HfApi, create_repo
        api = HfApi(token=token)

        repo_id = f"{repo_name}/{model_name}"
        try:
            create_repo(repo_id, token=token, exist_ok=True)
            print(f"  HuggingFace repo: https://huggingface.co/{repo_id}")
        except Exception as e:
            print(f"  Repo creation note: {e}")

        api.upload_folder(
            folder_path=local_path,
            repo_id=repo_id,
            token=token,
        )
        print(f"  Uploaded to HuggingFace: https://huggingface.co/{repo_id}")
        return True
    except Exception as e:
        print(f"  HuggingFace push error: {e}")
        return False


class TimeTracker:
    def __init__(self, max_hours=8.5):
        self.start_time = time.time()
        self.max_seconds = max_hours * 3600

    def elapsed(self):
        return time.time() - self.start_time

    def remaining(self):
        return max(0, self.max_seconds - self.elapsed())

    def expired(self):
        return self.remaining() <= 0

    def format(self, seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def status_str(self):
        return f"Elapsed: {self.format(self.elapsed())} | Remaining: {self.format(self.remaining())} | Budget: {self.format(self.max_seconds)}"

    def heartbeat(self, interval=300):
        now = time.time()
        if not hasattr(self, "_last_heartbeat") or now - self._last_heartbeat > interval:
            self._last_heartbeat = now
            print(f"  [HEARTBEAT] {self.status_str()}")
            return True
        return False
