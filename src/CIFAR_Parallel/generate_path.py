'''
Copyright (c) Mahsa Paknezhad, 2021
'''

from __future__ import print_function
import argparse
import torch
import random
import json
from util import *


def main(args):
    if not os.path.exists(args['inputfiles']%args['dataset']):
        os.mkdir(args['inputfiles']%args['dataset'])

    seed = args['seeds'][args['test_case']]
    print('seed', seed)

    # Use CUDA
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)

    for ses in range(args['num_tasks']):
        path = get_path(args['L'], args['M'], args['N'])
        print(str(path).replace('.', ','))
        np.save(args['inputfiles']%args['dataset'] + "/path_new_" + str(ses) + "_" + str(args['test_case']) + ".npy", path)

json_file = '../../scripts/test_case0_cifar100_parallel.json'
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', default=json_file, type=str, help='input json file')
    parsed = parser.parse_args()
    with open(parsed.json_file) as f:
        args = json.load(f)
    main(args)
