import os
import itertools

models = ['rgcn', 'rgat', 'hgt']
# datasets = ['freebase', 'igb-het', 'mag240m']
datasets = ['ogbn-mag']
cache_methods = ['heuristics']
embedding_sizes = [64]
predict_category = {
    'ogbn-mag': 'paper',
    'freebase': 'BOOK',
    'igb-het': 'paper',
    'mag240m': 'paper',
    'donor': 'Project'
}
number_of_classes = {
    'ogbn-mag': 349,
    'freebase': 8,
    'igb-het': 2983,
    'mag240m': 153,
    'donor': 2
}
batch_size = {
    'ogbn-mag': 128,
    'freebase': 128,
    'igb-het': 512,
    'mag240m': 128, 
    'donor': 128
}
ntypes_w_feats = {
    'ogbn-mag': ['paper'],
    'freebase': [],
    'igb-het': ['paper', 'author', 'institute', 'fos'],
    'mag240m': ['paper'],
    'donor': ['Project,Donor,Donation,Essay,School,Teacher,Resource']
}

for model, dataset, cache_method, embedding_size in itertools.product(models, datasets, cache_methods, embedding_sizes):
    print(f"Running {model} on {dataset}")
    cmd = f"./run.sh {model} {dataset} {predict_category[dataset]} {number_of_classes[dataset]} {batch_size[dataset]}"
    if len(ntypes_w_feats[dataset]) > 0:
        cmd += f" {','.join(ntypes_w_feats[dataset])}"
    cmd += f" {cache_method} {embedding_size}"
    os.system(cmd)
