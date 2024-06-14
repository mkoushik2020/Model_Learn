cd /content/ML_PROJECT


# Install the necessary libraries
!pip install tqdm gitpython

from git import Repo
from tqdm import tqdm
import os
import git
from git.remote import RemoteProgress

# Define a class to handle the progress reporting
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

# Function to clone a Git repository with progress display
def clone_with_progress(repo_url, clone_dir):
    if not os.path.exists(clone_dir):
        os.makedirs(clone_dir)
    progress = CloneProgress()
    Repo.clone_from(repo_url, clone_dir, progress=progress)

# Clone the repository with progress display
repo_url = 'https://github.com/mkoushik2020/Model_Learn.git'
clone_dir = '/content/ML_PROJECT'
clone_with_progress(repo_url, clone_dir)




pip install ultralytics


!python car_detection.py


!python train.py
