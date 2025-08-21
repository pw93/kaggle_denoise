# -*- coding: utf-8 -*-

import kaggle_env
import train
import os
import sys
import subprocess


def kaggle_init():
    # Detect if running in Kaggle
    is_kaggle = kaggle_env.detect_kaggle_runtime()
    print("is_kaggle: ",is_kaggle)

    if is_kaggle:

        def git_clone_or_pull(repo_url, target_dir):
            if not os.path.exists(target_dir):
                subprocess.run(['git', 'clone', repo_url, target_dir], check=True)
            else:
                subprocess.run(['git', '-C', target_dir, 'pull'], check=True)

        # Clone or pull repos
        git_clone_or_pull('https://github.com/pw93/kaggle_denoise.git', '/kaggle/working/code/kaggle_denoise')
        git_clone_or_pull('https://github.com/pw93/kaggle_utils.git', '/kaggle/working/code/kaggle_utils')

        # Add to sys.path
        paths_to_add = [
            '/kaggle/working/code/kaggle_denoise',
            '/kaggle/working/code/kaggle_utils',
            '/kaggle/working/code',
        ]
        for path in paths_to_add:
            if path not in sys.path:
                sys.path.append(path)
        print(sys.path)


kaggle_init()
train.train()
