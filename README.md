# Machine Learning Project Sunsky Software

<div style="text-align:center">
  <img src="http://www.sunskysoftware.com/images/logo-default-229x43.png" alt="Sunsky Software Logo">
</div>

## Create New Colab Notebook : [Click Here](https://colab.research.google.com/#create=true)

## Step 0: Change Runtime TO GPU
```
Runtime > Change Runtime > Select GPU
```

## Step 1: Navigate to Project Directory

Navigate to your project directory using the following command:
```
mkdir -p /content/Model_Learn
```

```
cd /content/Model_Learn
```

## Step 2: Install Required Packages

Install the necessary Python packages using pip:

```
pip install tqdm gitpython
```

## Step 3: Clone Git Repository with Progress Bar

Clone a Git repository with a progress bar implemented in Python:
```
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
clone_dir = '/content/Model_Learn'
clone_with_progress(repo_url, clone_dir)
```

## Step 4: Install Ultralytics Package

Install the Ultralytics package for your machine learning project:
```
pip install ultralytics
```

## Step 5: Run car_detection.py Script

Execute the car_detection.py script for car detection:
```
!python car_detection.py
```

Input File :
```
/content/Model_Learn/Input/bike.webp
```

## Step 6: Run train.py Script

Run the train.py script to train your machine learning model:
```
!python train.py
```

---

Replace repo_url, clone_dir, car_detection.py, and train.py with your specific repository URL, directory paths, and script names as per your project setup.

This README provides step-by-step instructions for setting up and running your machine learning project.
