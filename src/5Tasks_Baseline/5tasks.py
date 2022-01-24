from __future__ import print_function

import json
import time
import torch.nn.parallel
from utils import Bar, mkdir_p
import argparse
import torch
import sys
from net_utils import get_network
import random
from learner import Learner
from util import *
import fine_grained_dataset as dataset





def main(args):
    checkpoint = args['checkpoint']
    args['inputfiles'] = args['inputfiles']%args['dataset']
    for task_num in range(int(args["num_tasks"])):
        args['task_num'] = task_num
        args['checkpoint'] = checkpoint%(args['dataset'],int(args['test_case']))
        seed = args['seeds'][args['test_case']]
        print('seed', seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if args['use_cuda']:
            torch.cuda.manual_seed_all(seed)

        model = get_network(args)
        print('model is generated')
        loaded_checkpoint = torch.load(os.path.join(args['inputfiles'], 'init_model_new_%s.pth.tar' % str(args['test_case'])))
        print('checkpoint is loaded')
        model.load_state_dict(loaded_checkpoint)
        print('restored checkpoint successfully')
        model.train()
        if args['use_cuda']:
            model = model.cuda()

        if args['use_imagenet_pretrained']:
            if args['arch'] == 'resnet50' or args['arch'] == 'resnet18':
                state_dict = torch.load(os.path.join(args['modelfile'], args['arch'] + '.pth'))
                cur_state_dict = model.state_dict()
                for name, param in state_dict.items():
                    if 'fc' not in name and 'bn' not in name and 'downsample.1' not in name:
                        parts = name.split('.', 1)
                        for j in range(args['M']):
                            cur_state_dict[parts[0] + str(j) + '.' + parts[1]].copy_(param)
                    elif 'bn' in name:
                        parts = name.split('.')
                        if len(parts) == 2:
                            for j in range(args['M']):
                                for k in range(args['num_tasks']):
                                    cur_state_dict[parts[0] + str(j) + str(k) + '.' + parts[1]].copy_(param)
                        else:
                            for j in range(args['M']):
                                for k in range(args['num_tasks']):
                                    cur_state_dict[
                                        parts[0] + str(j) + '.' + parts[1] + '.' + parts[2] + str(k) + '.' + parts[
                                            3]].copy_(param)
                    elif 'downsample.1' in name:
                        parts = name.split('.')
                        for j in range(args['M']):
                            for k in range(args['num_tasks']):
                                cur_state_dict[
                                    parts[0] + str(j) + '.' + parts[1] + '.' + parts[2] + '.' + parts[3] + '.' + str(
                                        k) + '.' + parts[4]].copy_(param)
            else:
                print(
                    "Currently, we didn't define the mapping of {} between imagenet pretrained weight and our model".format(
                        args['arch']))
                sys.exit(5)

        print('    Total params: %.2fM' % (sum(p.numel() for n,p in model.named_parameters()) / 1000000.0))

        if not os.path.isdir(args['checkpoint']):
            mkdir_p(args['checkpoint'])

        args['savepoint'] = args['checkpoint']
        with open(os.path.join(args["checkpoint"], "setting.json"), 'w') as json_file:
            json.dump(args, json_file, indent=4, sort_keys=True)

        path = np.load(args['inputfiles'] + "/path_new_" + str(task_num) + "_" + str(args['test_case']) + ".npy")
        train_path = path.copy()
        infer_path = path.copy()

        print('Starting with session {:d}'.format(task_num))
        print('test case : ' + str(args['test_case']))
        print('#################################################################################')
        print("path\n", path)
        print("train_path\n", train_path)

        cur_dataset = args['datasets'][task_num]
        if 'cropped' in cur_dataset:
            train_loader = dataset.train_loader_cropped(os.path.join(args['datasets_dir'], cur_dataset, 'train'),
                                                        args['train_batch'], num_workers=args['workers'])
            val_loader = dataset.val_loader_cropped(os.path.join(args['datasets_dir'], cur_dataset, 'test'),
                                                    args['train_batch'], num_workers=args['workers'])
        else:
            train_loader = dataset.train_loader(os.path.join(args['datasets_dir'], cur_dataset, 'train'), args['train_batch'], num_workers=args['workers'])
            val_loader = dataset.val_loader(os.path.join(args['datasets_dir'], cur_dataset, 'test'), args['train_batch'], num_workers=args['workers'])

        class_per_task = args['num_classes'][task_num]

        logit_init_ind = 0
        for s in range(0, task_num):
            logit_init_ind += args['num_classes'][s]

        main_learner = Learner(model=model, args=args, trainloader=train_loader,
                               testloader=val_loader,  class_per_task=class_per_task,
                               logit_init_ind=logit_init_ind, use_cuda=args['use_cuda'], path=path,
                               train_path=train_path, infer_path=infer_path)
        main_learner.learn()

        cfmat = main_learner.get_confusion_matrix(infer_path)
        np.save(args['checkpoint'] + "/confusion_matrix_" + str(task_num) + "_" + str(args['test_case']) + ".npy", cfmat)

        print('done with session {:d}'.format(task_num))
        print('#################################################################################')
        while (1):
            if (is_all_done(task_num, args['num_epochs'], args['checkpoint'])):
                break
            else:
                time.sleep(10)


json_file = '../../scripts/test_case0_5tasks_baseline.json'
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', default=json_file, type=str, help='input json file')
    parsed = parser.parse_args()
    with open(parsed.json_file) as f:
        args = json.load(f)
    main(args)
