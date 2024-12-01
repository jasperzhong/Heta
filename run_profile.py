import os

datasets = ['ogbn-mag', 'freebase', 'donor', 'igb-het', 'mag240m']

for dataset in datasets:
    cmd = f"python miss_penalty.py --dataset {dataset} 2>&1 >> miss_penalty.log"
    os.system(cmd)
