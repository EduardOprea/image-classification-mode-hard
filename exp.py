import os
def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

ensure_dir_exists("results/testceva/checkpoints")