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
import json
import torch
import random
import argparse
from cifar_net import cifar_net
from learner import Learner
from util import *
from cifar_dataset import CIFAR100, CIFAR10


def main(args):
    seed = args["seeds"][int(args["test_case"])]
    print('seed', seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if args["use_cuda"]:
        torch.cuda.manual_seed_all(seed)

    args['inputfiles'] = args['inputfiles'] % (args['dataset'])
    args['checkpoint'] = args['checkpoint'] % (args['dataset'], args['test_case'])
    args['labels_data'] = args['labels_data']% (args['dataset'], args['test_case'])
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

    sessions = list(range(args['num_tasks']))

    inds_all_sessions = pickle.load(open(os.path.join(args['inputfiles'], args['labels_data']), 'rb'))

    # load the assigned path (sequence of modules) to the tasks (or ses here)
    train_paths = []
    for ses in sessions:
        path = np.load(args['inputfiles'] + "/path_new_" + str(ses) + "_" + str(args['test_case']) + ".npy")
        print("train_path %d\n" % ses, path)
        train_paths.append(path)

    print('test case : ' + str(args['test_case']))
    print('#################################################################################')

    trainloaders = []
    testloaders = []
    alllabels = []
    logit_init_inds = [0]
    class_per_tasks = []
    logit_init_ind = 0
    args['dataset']=[]

    # load the tasks. This includes the class labels and the index of the images in the dataset for each task
    for ses in sessions:
        ind_this_session = inds_all_sessions[ses]
        labels = ind_this_session['labels'][0]
        ind_trn = ind_this_session['curent']
        ind_tst = ind_this_session['test']
        class_per_tasks.append(len(labels))
        logit_init_ind += len(labels)
        logit_init_inds.append(logit_init_ind)

        if 'name' in ind_this_session.keys():
            args['dataset'].append(ind_this_session['name'])
        else:
            args['dataset'].append('cifar100')

        # generate the train and test dataloaders for each task using the index of images assigned to the task
        if args['dataset'][-1] == 'cifar100':

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

        trainloaders.append(trainloader)
        testloaders.append(testloader)
        alllabels.append(labels)

    main_learner = Learner(model=model, args=args, sessions=sessions, trainloaders=trainloaders,
                           testloaders=testloaders, labels=alllabels, class_per_tasks=class_per_tasks, logit_init_inds= logit_init_inds,
                           use_cuda=args['use_cuda'], train_paths=train_paths)

    # train the base network on the defined tasks in parallel
    main_learner.learn()

    for ses in sessions:
        cfmat = main_learner.get_confusion_matrix(ses)
        np.save(args['checkpoint'] + "/confusion_matrix_" + str(ses) + "_" + str(args['test_case']) + ".npy", cfmat)

        print('confusion matrix for task {:d}'.format(ses))
        print('#################################################################################')
        while (1):
            if (is_all_done(ses, args['epochs'], args['checkpoint'])):
                break
            else:
                time.sleep(10)



json_file = '../../scripts/test_case0_cifar100_parallel.json'
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', default=json_file, type=str, help='input json file')
    parsed = parser.parse_args()
    with open(parsed.json_file) as f:
        args = json.load(f)
    main(args)
