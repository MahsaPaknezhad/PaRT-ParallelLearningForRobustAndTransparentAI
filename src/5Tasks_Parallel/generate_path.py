from __future__ import print_function
import json
import torch
import random
import numpy as np
from util import *
import argparse



def main(args):
	if not os.path.exists(args['inputfiles']):
		os.mkdir(args['inputfiles'])

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
		np.save(args['inputfiles'] + "/path_new_" + str(ses) + "_" + str(args['test_case']) + ".npy", path)


json_file = '../../scripts/test_case0_5tasks_parallel.json'
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', default=json_file, type=str, help='input json file')
    parsed = parser.parse_args()
    with open(parsed.json_file) as f:
        args = json.load(f)
    main(args)

