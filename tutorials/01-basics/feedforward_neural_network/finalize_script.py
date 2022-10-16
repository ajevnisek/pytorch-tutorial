import os
import shutil
from common import RESULTS_DIR


STORAGE_DIR = '/storage/amir/runai'


os.makedirs(STORAGE_DIR, exist_ok=True)
shutil.copytree(RESULTS_DIR, STORAGE_DIR,  dirs_exist_ok=True)
