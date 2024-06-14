# Step 0
cd /content/ML_PROJECT



# Step 1
!pip install tqdm gitpython



# Step 2

from git import Repo
from tqdm import tqdm
import os
import git
from git.remote import RemoteProgress

class CloneProgress(RemoteProgress):
    def __init__(self):
        super().__init__()
        self.pbar = None
    
    def update(self, op_code, cur_count, max_count=None, message=''):
        if self.pbar is None:
            self.pbar = tqdm(total=max_count, unit='objects', unit_scale=True)
        self.pbar.n = cur_count
        self.pbar.refresh()
        if cur_count >= self.pbar.total:
            self.pbar.close()
            self.pbar = None
            
def clone_with_progress(repo_url, clone_dir):
    if not os.path.exists(clone_dir):
        os.makedirs(clone_dir)
    progress = CloneProgress()
    Repo.clone_from(repo_url, clone_dir, progress=progress)

repo_url = 'https://github.com/mkoushik2020/Model_Learn.git'
clone_dir = '/content/ML_PROJECT'
clone_with_progress(repo_url, clone_dir)



# Step 3

pip install ultralytics



# Step 4
!python car_detection.py



# Step 5
!python train.py
