'''
Copyright (c) Mahsa Paknezhad, 2021
'''
from net_utils import get_network
import torch
import os
import argparse
import json
import random
import numpy as np


def main(args):
	if not os.path.exists(args['inputfiles']%args['dataset']):
		os.mkdir(args['inputfiles']%args['dataset'])

	seed = args['seeds'][args['test_case']]
	print('seed', seed)
	torch.manual_seed(seed)
	random.seed(seed)
	np.random.seed(seed)
	if args['use_cuda']:
		torch.cuda.manual_seed_all(seed)

	net = get_network(args)
	torch.save(net.state_dict(), os.path.join(args['inputfiles']%args['dataset'], 'init_model_new_' + str(args['test_case']) + '.pth.tar'))


json_file = '../../scripts/test_case0_5tasks_baseline.json'
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', default=json_file, type=str, help='input json file')
    parsed = parser.parse_args()
    with open(parsed.json_file) as f:
        args = json.load(f)
    main(args)