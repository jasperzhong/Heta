import os
import itertools

models = ['rgcn']
# datasets = ['freebase', 'igb-het', 'mag240m']
datasets = ['igb-het']
part_methods = ['random']
cache_methods = ['heuristics']
embedding_sizes = [64]
fanouts = ['25,20']

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
    'ogbn-mag': 64,
    'freebase': 64,
    'igb-het': 256,
    'mag240m': 64,
    'donor': 32
}

for model, dataset, part_method, cache_method, embedding_size, fanout in itertools.product(models, datasets, part_methods, cache_methods, embedding_sizes, fanouts):
    print(f"Running {model} on {dataset} with {part_method}")
    os.system(f"./run.sh {model} {dataset} {part_method} {predict_category[dataset]} {number_of_classes[dataset]} {batch_size[dataset]} {cache_method} {embedding_size} {fanout}")
