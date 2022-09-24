import os


try:
    RESULTS_DIR = '/results'
    os.makedirs(RESULTS_DIR, exist_ok=True)
except:
    RESULTS_DIR = '/tmp/results'
    os.makedirs(RESULTS_DIR, exist_ok=True)

