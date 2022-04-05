'''
Copyright (c) Mahsa Paknezhad, 2021
'''

from __future__ import print_function
import time
import pickle
import torch.nn.parallel
import torch.utils.data as data
import torchvision.transforms as transforms
from utils import Bar, mkdir_p
import argparse
import torch
import random
from cifar_net import cifar_net
from learner import Learner
from util import *
from cifar_dataset import CIFAR100, CIFAR10
import json



def main(args):
    inputfiles = args['inputfiles']
    checkpoint = args['checkpoint']
    labels_data = args['labels_data']
    dataset = args['dataset']
    for task_num in range(int(args["num_tasks"])):
        args['task_num'] = task_num
        seed = args['seeds'][args['test_case']]
        print('seed', seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if args['use_cuda']:
            torch.cuda.manual_seed_all(seed)

        args['inputfiles'] = inputfiles % (dataset)
        args['checkpoint'] = checkpoint % (dataset, args['test_case'])
        args['labels_data'] = labels_data % (dataset, args['test_case'])
        print("SHUFFLED DATA: ", args['labels_data'])

        if args['use_cuda']:
            device = 'cuda'
            torch.cuda.manual_seed_all(seed)
        else:
            device = 'cpu'

        # Create an object of the base network
        model = cifar_net(args)
        model.load_state_dict(torch.load(os.path.join(args['inputfiles'], 'init_model_new_%s.pth.tar' % str(args['test_case'])),
                                         map_location=torch.device(device)))
        model.train()
        if args['use_cuda']:
            model = model.cuda()

        if not os.path.isdir(args['checkpoint']):
            mkdir_p(args['checkpoint'])

        args['savepoint'] = args['checkpoint']
        with open(os.path.join(args["checkpoint"], "setting.json"), 'w') as json_file:
            json.dump(args, json_file, indent=4, sort_keys=True)

        # Specify the transformation functions for tasks defined on CIFAR10 and CIFAR100 datasets
        cifar10_transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomPerspective(p=0.75, distortion_scale=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        cifar10_transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        cifar100_transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomPerspective(p=0.75, distortion_scale=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.486, 0.44), (0.267, 0.256, 0.276)),
        ])

        cifar100_transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.486, 0.44), (0.267, 0.256, 0.276)),
        ])

        inds_all_sessions = pickle.load(open(os.path.join(args['inputfiles'], args['labels_data']), 'rb'))

        # load the assigned path (sequence of modules) to the input task
        path = np.load(args['inputfiles'] + "/path_new_" + str(args['task_num']) + "_" + str(args['test_case']) + ".npy")
        train_path = path.copy()
        infer_path = path.copy()

        print('Starting with task {:d}'.format(args['task_num']))
        print('test case : ' + str(args['test_case']))
        print('#################################################################################')
        print("path\n", path)
        print("train_path\n", train_path)

        # load the tasks. This includes the class labels and the index of the images in the dataset for each task
        logit_init_ind = 0
        for i in range(args['task_num']):
            this_session = inds_all_sessions[i]
            this_labels = this_session['labels'][0]
            logit_init_ind += len(this_labels)

        ind_this_session = inds_all_sessions[args['task_num']]
        labels = ind_this_session['labels'][0]
        ind_trn = ind_this_session['curent']
        ind_tst = ind_this_session['test']
        class_per_task = len(labels)
        if 'name' in ind_this_session.keys():
            args['dataset'] = ind_this_session['name']
        else:
            args['dataset'] = 'cifar100'

        # generate the train and test dataloaders for each task using the index of images assigned to the task
        if args['dataset'] == 'cifar100':

            dataloader = CIFAR100
            trainset = dataloader(root=args['datasets_dir'], train=True, download=True, transform=cifar100_transform_train, ind=ind_trn)
            trainloader = data.DataLoader(trainset, batch_size=args['train_batch'], shuffle=True, num_workers=args['workers'])
            testset = dataloader(root=args['datasets_dir'], train=False, download=False, transform=cifar100_transform_test, ind=ind_tst)
            testloader = data.DataLoader(testset, batch_size=args['test_batch'], shuffle=False, num_workers=args['workers'])

        else:

            dataloader = CIFAR10
            trainset = dataloader(root=args['datasets_dir'], train=True, download=True, transform=cifar10_transform_train, ind=ind_trn)
            trainloader = data.DataLoader(trainset, batch_size=args['train_batch'], shuffle=True, num_workers=args['workers'])
            testset = dataloader(root=args['datasets_dir'], train=False, download=False, transform=cifar10_transform_test, ind=ind_tst)
            testloader = data.DataLoader(testset, batch_size=args['test_batch'], shuffle=False, num_workers=args['workers'])

        main_learner = Learner(model=model, args=args, trainloader=trainloader,
                               testloader=testloader, labels=labels, class_per_task=class_per_task,
                               logit_init_ind=logit_init_ind, use_cuda=args['use_cuda'], path=path,
                               train_path=train_path, infer_path=infer_path)

        # train the base network from scratch on the input task
        main_learner.learn()

        cfmat = main_learner.get_confusion_matrix(infer_path)
        np.save(args['checkpoint'] + "/confusion_matrix_" + str(args['task_num']) + "_" + str(args['test_case']) + ".npy", cfmat)

        print('confusion matrix for task {:d}'.format(args['task_num']))
        print('#################################################################################')
        while (1):
            if (is_all_done(args['task_num'], args['epochs'], args['checkpoint'])):
                break
            else:
                time.sleep(10)



json_file = '../../scripts/test_case0_cifar10_100_baseline.json'
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', default=json_file, type=str, help='input json file')
    parsed = parser.parse_args()
    with open(parsed.json_file) as f:
        args = json.load(f)
    main(args)
